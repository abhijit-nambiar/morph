#!/usr/bin/env python3
"""Compare face, frame grid and combined models on independent face-light targets."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,face_light_experiment as face,grid_light,hybrid_face_real_compare as hybrid


def independent_mask(shape,boxes):
    # Elliptical plateau with a smooth cosine feather; differs from fitted Gaussian masks.
    h,w=shape[:2];yy=(np.arange(h,dtype=np.float32)+.5)/h;xx=(np.arange(w,dtype=np.float32)+.5)/w;out=np.zeros((h,w),dtype=np.float32)
    for b in boxes:
        cx=b['x']+b['width']*.5;cy=b['y']+b['height']*.5
        distance=np.sqrt(((xx[None,:]-cx)/(b['width']*1.2))**2+((yy[:,None]-cy)/(b['height']*1.1))**2)
        mask=np.where(distance<=.35,1.,np.where(distance>=1,0.,.5+.5*np.cos(np.pi*(distance-.35)/.65)))
        out=np.maximum(out,mask)
    return out

def target(x,boxes,settings,strength):return np.clip(morph.pillow_soft_target(x,'neutral',settings)+strength*independent_mask(x.shape,boxes)[...,None]*x*(1-x),0,1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('detections');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--regularization',type=float,default=.0001);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));detections={str(Path(r['path']).resolve()):r for r in json.loads(Path(a.detections).read_text())};settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[];skipped=[]
    for g in groups:
        rep=g['representative'];d=detections.get(str(Path(rep).resolve()))
        if not d or d.get('error') or not d['faces']:skipped.append({'scene':g['id'],'reason':'missing representative detection'});continue
        x=morph.load(rep,a.size);others=[p for p in g['photos'] if p!=rep]
        for strength in (-.3,0.,.3):
            y=target(x,d['faces'],settings,strength);wg,_=grid_light.fit_with_metrics(x,y,regularization=a.regularization);wf=face.fit(x,y,d['faces'],a.regularization);wh=hybrid.fit(x,y,d['faces'],a.regularization)
            for path in others:
                detection=detections.get(str(Path(path).resolve()))
                if not detection or detection.get('error') or not detection['faces']:
                    if strength==0:skipped.append({'scene':g['id'],'photo':str(path),'reason':'missing detection'})
                    continue
                xh=morph.load(path,a.size);wanted=target(xh,detection['faces'],settings,strength);boxes=detection['faces']
                rows.append({'scene':g['id'],'photo':str(path),'strength':strength,'source_rmse':face.rmse(xh,wanted),'grid_rmse':face.rmse(grid_light.render(xh,wg),wanted),'face_rmse':face.rmse(face.render(xh,boxes,wf),wanted),'hybrid_rmse':face.rmse(np.clip(xh+hybrid.features(xh,boxes)@wh,0,1),wanted)})
        print(g['id'],len(others),'held-out',flush=True)
    if not rows:raise ValueError('No held-out photos')
    summary={}
    for strength in (-.3,0.,.3):
        sub=[r for r in rows if r['strength']==strength];g=np.array([r['grid_rmse'] for r in sub]);f=np.array([r['face_rmse'] for r in sub]);h=np.array([r['hybrid_rmse'] for r in sub]);summary[str(strength)]={'heldout':len(sub),'face_lower_than_grid':int(np.sum(f<g)),'hybrid_lower_than_grid':int(np.sum(h<g)),'grid_mean_rmse':float(g.mean()),'face_mean_rmse':float(f.mean()),'hybrid_mean_rmse':float(h.mean())}
    morph.dump({'scope':'One album, synthetic face-relative lighting uses an ellipse with cosine feather rather than the fitted Gaussian mask. Both still use the same detector boxes, so detection accuracy is not independently assessed; no real edit or photographic quality validation.','size':a.size,'regularization':a.regularization,'summary':summary,'skipped':skipped,'rows':rows},a.out)
if __name__=='__main__':main()
