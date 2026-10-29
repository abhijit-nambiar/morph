#!/usr/bin/env python3
"""Estimate exposure/color corrections from geometrically matched scene content.

Only the supplied scene reference originals are calibration inputs. Other clean
counterparts and synthetic perturbation settings are not accepted by this tool.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import morph
import gamut_renderer


def logit(x):
    x = np.clip(x, .00001, .99999)
    return np.log(x / (1-x))


def render(x, correction):
    # A single contrast scale plus independent RGB offsets. Endpoints are
    # preserved because lost black/white information cannot be reconstructed.
    values = logit(np.clip(x, .00001, .99999))
    z = values * correction['scale'] + np.asarray(correction['offset'], dtype=np.float32)
    result = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return np.where(x <= 0, 0, np.where(x >= 1, 1, result))


def gamut_recipe(correction):
    red,green,blue=correction['offset']
    common=(red+blue)/2
    recipe={'exposure':float(common/np.log(2)),
            'temperature':float((red-blue)/2.6),
            'tint':float((green-common)/1.3),
            'contrast':float(correction['scale']-1)}
    gamut_renderer.vec(recipe)
    return recipe


def extract(path, size):
    image = morph.load(path, size)
    gray = cv2.cvtColor(np.uint8(image*255+.5), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    points, descriptors = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.02).detectAndCompute(gray, None)
    return image, points, descriptors


def patch(image, point):
    x, y = np.round(point.pt).astype(int)
    radius = int(np.clip(point.size*.22, 2, 8))
    h, w = image.shape[:2]
    region = image[max(0,y-radius):min(h,y+radius+1), max(0,x-radius):min(w,x+radius+1)]
    return np.median(region.reshape(-1, 3), axis=0)


def correspondences(left, right):
    a, ka, da = left
    b, kb, db = right
    if da is None or db is None or min(len(da), len(db)) < 12:
        return None
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    matches = [m for pair in matcher.knnMatch(da, db, k=2) if len(pair)==2
               for m, n in [pair] if m.distance < .7*n.distance]
    if len(matches) < 16:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    homography, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3)
    if mask is None:
        return None
    good = [m for m, ok in zip(matches, mask.ravel()) if ok]
    if len(good) < 14 or len(good)/len(matches) < .2:
        return None
    # Compare the same spatial samples after geometric alignment. Sparse patch
    # centers alone were unstable under JPEG re-encoding and SIFT scale changes.
    h,w=a.shape[:2]
    warped=cv2.warpPerspective(b,np.linalg.inv(homography),(w,h))
    coverage=cv2.warpPerspective(np.ones(b.shape[:2],dtype=np.float32),np.linalg.inv(homography),(w,h))>.999
    left=cv2.GaussianBlur(a,(5,5),1)
    right=cv2.GaussianBlur(warped,(5,5),1)
    gray_a=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
    gray_b=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
    ax=cv2.Sobel(gray_a,cv2.CV_32F,1,0);ay=cv2.Sobel(gray_a,cv2.CV_32F,0,1)
    bx=cv2.Sobel(gray_b,cv2.CV_32F,1,0);by=cv2.Sobel(gray_b,cv2.CV_32F,0,1)
    na=np.sqrt(ax*ax+ay*ay);nb=np.sqrt(bx*bx+by*by)
    agreement=(ax*bx+ay*by)/np.maximum(na*nb,1e-6)
    valid=coverage&(na>.01)&(nb>.01)&(agreement>.92)
    valid[:5]=False;valid[-5:]=False;valid[:,:5]=False;valid[:,-5:]=False
    yy,xx=np.nonzero(valid)
    samples=[]
    if len(yy)>=60:
        selected=np.linspace(0,len(yy)-1,min(4000,len(yy))).astype(int)
        samples.append((left[yy[selected],xx[selected]],right[yy[selected],xx[selected]],'aligned-overlap'))
    samples.append((np.stack([patch(a,ka[m.queryIdx]) for m in good]),
                    np.stack([patch(b,kb[m.trainIdx]) for m in good]),'keypoint-patches'))
    for aa,bb,sampling in samples:
        result=fit_observations(aa,bb)
        if result is not None:
            aa,bb,coef,error=result
            return aa,bb,{'matches':len(matches),'geometric_inliers':len(good),
                          'photometric_inliers':len(aa),'pair_scale':float(coef[0]),
                          'pair_residual':error,'sampling':sampling}
    return None


def fit_observations(aa,bb):
    valid=np.all((aa>.025)&(aa<.975)&(bb>.025)&(bb<.975),axis=1)
    aa,bb=logit(aa[valid]),logit(bb[valid])
    if len(aa)<12:return None
    design=np.zeros((len(aa)*3,4));design[:,0]=aa.ravel()
    design[:,1:]=np.tile(np.eye(3),(len(aa),1));rhs=bb.ravel()
    weight=np.ones(len(rhs))
    for _ in range(8):
        coef=np.linalg.lstsq(design*weight[:,None]**.5,rhs*weight**.5,rcond=None)[0]
        residual=design@coef-rhs
        spread=max(.025,1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        weight=np.minimum(1,1.5*spread/np.maximum(np.abs(residual),1e-6))
    errors=np.sqrt(np.mean(residual.reshape(-1,3)**2,axis=1))
    keep=errors<max(.08,float(np.median(errors))*2.5)
    if sum(keep)<12 or not .65<coef[0]<1.4:return None
    return aa[keep],bb[keep],coef,float(np.median(errors[keep]))


def solve_scene(paths, anchor, features):
    # All input photos may be perturbed, including the representative. The
    # separate intended reference anchors the graph's otherwise ambiguous tone.
    nodes = list(paths) + [anchor]
    root = len(paths)
    pairs = {(i, root) for i in range(root)}
    pairs.update((i,j) for i in range(root) for j in range(i+1,min(root,i+4)))
    edges = []
    for i, j in sorted(pairs):
        result = correspondences(features[nodes[i]], features[nodes[j]])
        if result is not None:
            aa, bb, info = result
            edges.append((i,j,aa,bb,info))
    connected = {root}
    while True:
        updated = connected | {i for i,j,*_ in edges if j in connected} | {j for i,j,*_ in edges if i in connected}
        if updated == connected:
            break
        connected = updated
    active = sorted(connected-{root})
    lookup = {node:k for k,node in enumerate(active)}
    designs, targets, base_weights = [], [], []
    for i,j,aa,bb,info in edges:
        if i not in connected or j not in connected:
            continue
        # Limit each edge's contribution, so repeated textures do not dominate.
        idx = np.linspace(0,len(aa)-1,min(300,len(aa))).astype(int)
        aa, bb = aa[idx], bb[idx]
        design = np.zeros((len(aa)*3,len(active)*4)); rhs=np.zeros(len(aa)*3)
        for node, samples, sign in ((i,aa,1),(j,bb,-1)):
            if node == root:
                rhs -= sign*samples.ravel()
            else:
                start = lookup[node]*4
                design[:,start] = sign*samples.ravel()
                design[:,start+1:start+4] = sign*np.tile(np.eye(3),(len(samples),1))
        designs.append(design);targets.append(rhs)
        base_weights.extend([1/max(.04,info['pair_residual'])] * len(rhs))
    results = {path:{'scale':1.,'offset':[0.,0.,0.], 'status':'review-unconnected'} for path in paths}
    if active:
        design=np.concatenate(designs);rhs=np.concatenate(targets);base=np.asarray(base_weights);weight=base.copy()
        prior=np.tile([1.,0.,0.,0.],len(active))
        for _ in range(8):
            normal=design.T@(design*weight[:,None])+.05*np.eye(design.shape[1])
            coef=np.linalg.solve(normal,design.T@(rhs*weight)+.05*prior)
            residual=design@coef-rhs
            spread=max(.03,1.4826*float(np.median(np.abs(residual-np.median(residual)))))
            weight=base*np.minimum(1,1.5*spread/np.maximum(np.abs(residual),1e-6))
        for node in active:
            value=coef[lookup[node]*4:lookup[node]*4+4]
            safe=.7<float(value[0])<1.35 and np.max(np.abs(value[1:]))<.9
            results[nodes[node]]={'scale':float(value[0]) if safe else 1.,
                                'offset':value[1:].tolist() if safe else [0.,0.,0.],
                                'proposed_scale':float(value[0]),'proposed_offset':value[1:].tolist(),
                                'status':'estimated' if safe else 'review-outside-bounds'}
    diagnostics=[]
    identity={'scale':1.,'offset':[0.,0.,0.]}
    for i,j,aa,bb,info in edges:
        ci=identity if i==root else results[nodes[i]]
        cj=identity if j==root else results[nodes[j]]
        corrected_a=aa*ci['scale']+np.asarray(ci['offset'])
        corrected_b=bb*cj['scale']+np.asarray(cj['offset'])
        diagnostics.append({'left':nodes[i],'right':nodes[j],**info,
                            'before_logit_rmse':float(np.sqrt(np.mean((aa-bb)**2))),
                            'after_logit_rmse':float(np.sqrt(np.mean((corrected_a-corrected_b)**2)))})
    return results, diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups');parser.add_argument('--references',required=True)
    parser.add_argument('--size',type=int,default=1000);parser.add_argument('--out',required=True)
    parser.add_argument('--corrections-out',help='Standard photo-correction JSON for morph.py apply')
    args=parser.parse_args()
    if not 256<=args.size<=1600:
        raise ValueError('size must be 256–1600')
    groups=morph.validate_groups(json.loads(Path(args.groups).read_text()))
    references=json.loads(Path(args.references).read_text())
    if set(references)!={g['id'] for g in groups}:
        raise ValueError('Exactly one reference is required per scene')
    paths=[p for g in groups for p in g['photos']]
    inputs=[args.groups,args.references]+paths+list(references.values())
    outputs=[args.out]+([args.corrections_out] if args.corrections_out else [])
    morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False)
    before={p:morph.sha256_file(p) for p in dict.fromkeys(paths+list(references.values()))}
    cv2.setRNGSeed(0);cv2.setNumThreads(1)
    features={p:extract(p,args.size) for p in dict.fromkeys(paths+list(references.values()))}
    scenes=[];corrections={}
    for group in groups:
        result,edges=solve_scene(group['photos'],references[group['id']],features)
        corrections.update(result);scenes.append({'scene':group['id'],'reference':references[group['id']],'edges':edges})
        print(group['id'],len(edges),'usable edges;',sum(r['status']=='estimated' for r in result.values()),'/',len(result),'estimated',flush=True)
    if any(morph.sha256_file(p)!=h for p,h in before.items()):
        raise ValueError('Source changed during normalization')
    recipes={}
    for group in groups:
        recipes[group['id']]={}
        for path in group['photos']:
            try:recipe=gamut_recipe(corrections[path])
            except ValueError:
                corrections[path]={'scale':1.,'offset':[0.,0.,0.],'status':'review-outside-renderer-bounds'}
                recipe={}
            recipes[group['id']][Path(path).name]=recipe
    morph.dump({'kind':'matched-scene-normalization-v1','scope':'Geometrically verified SIFT patch matches and robust '
                'logit-color fitting to one intended reference per scene. No other clean counterparts, '
                'perturbation parameters or evaluation targets are used. Weakly connected photos require review. '
                'Reference exposure/color defines the desired scene baseline; shared style fitting provides '
                'the subsequent album look. This does not recover clipped detail or local illumination changes.',
                'size':args.size,'source_manifest':before,
                'scenes':scenes,'corrections':corrections},args.out)
    if args.corrections_out:morph.dump({'photos':recipes},args.corrections_out)


if __name__=='__main__':main()
