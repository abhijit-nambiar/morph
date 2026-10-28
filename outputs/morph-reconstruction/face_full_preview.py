#!/usr/bin/env python3
"""Experimental full-resolution face-relative synthetic reconstruction preview."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color
import face_light_experiment as face
import face_independent_experiment as independent


def tile_mask(shape,boxes,row_start,full_shape,mode):
    h,w=shape[:2];yy=(np.arange(row_start,row_start+h,dtype=np.float32)+.5)/full_shape[0];xx=(np.arange(w,dtype=np.float32)+.5)/full_shape[1];result=np.zeros((h,w),dtype=np.float32)
    for b in boxes:
        cx=b['x']+b['width']*.5;cy=b['y']+b['height']*.5
        if mode=='model':
            sx=max(b['width']*.8,.025);sy=max(b['height']*.8,.025);m=np.exp(-.5*(((xx[None,:]-cx)/sx)**2+((yy[:,None]-cy)/sy)**2))
        else:
            distance=np.sqrt(((xx[None,:]-cx)/(b['width']*1.2))**2+((yy[:,None]-cy)/(b['height']*1.1))**2)
            m=np.where(distance<=.35,1.,np.where(distance>=1,0.,.5+.5*np.cos(np.pi*(distance-.35)/.65)))
        result=np.maximum(result,m)
    return result


def model_tile(x,w,boxes,start,shape):
    m=tile_mask(x.shape,boxes,start,shape,'model');return np.clip(x+poly_color.features(x)@w[:10]+(m[...,None]*x*(1-x))@w[10:],0,1)

def target_tile(x,boxes,settings,strength,start,shape):
    m=tile_mask(x.shape,boxes,start,shape,'target');return np.clip(morph.pillow_soft_target(x,'neutral',settings)+strength*m[...,None]*x*(1-x),0,1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('detections');p.add_argument('settings');p.add_argument('--scene',required=True);p.add_argument('--photo',required=True,help='Full path to a non-representative photo');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=320);p.add_argument('--strength',type=float,default=.3);a=p.parse_args()
    if not np.isfinite(a.strength) or not -.5<=a.strength<=.5:raise ValueError('Strength must be between -.5 and .5')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));g=next((g for g in groups if g['id']==a.scene),None)
    if not g:raise ValueError('Unknown scene')
    rep=str(Path(g['representative']).resolve());photo=str(Path(a.photo).resolve());members={str(Path(p).resolve()) for p in g['photos']}
    if photo not in members or photo==rep:raise ValueError('Photo must be a non-representative scene member')
    lookup={str(Path(r['path']).resolve()):r for r in json.loads(Path(a.detections).read_text())}
    for path in (rep,photo):
        if path not in lookup or lookup[path].get('error') or not lookup[path]['faces']:raise ValueError('Missing face detection: '+path)
    settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));x=morph.load(rep,a.size);y=independent.target(x,lookup[rep]['faces'],settings,a.strength);w=face.fit(x,y,lookup[rep]['faces'])
    out=Path(a.out);outputs=[out/'report.json']+[out/f'{label}-{kind}.jpg' for label in ('representative','heldout') for kind in ('target','reconstruction')]
    morph.protect_inputs([a.groups,a.detections,a.settings,rep,photo],outputs);morph.protect_existing(outputs,False)
    rows=[]
    for label,path in (('representative',rep),('heldout',photo)):
        boxes=lookup[path]['faces'];sample=morph.load(path,a.size);want=independent.target(sample,boxes,settings,a.strength);estimate=face.render(sample,boxes,w)
        full=morph.load(path)
        def target_renderer(tile,unused):return target_tile(tile,boxes,settings,a.strength,0,tile.shape)
        target_renderer.render_tile=lambda tile,unused,start,shape:target_tile(tile,boxes,settings,a.strength,start,shape)
        def fitted_renderer(tile,weights):return model_tile(tile,weights,boxes,0,tile.shape)
        fitted_renderer.render_tile=lambda tile,weights,start,shape:model_tile(tile,weights,boxes,start,shape)
        morph.save_render(full,w,out/f'{label}-target.jpg',renderer=target_renderer,source=path)
        morph.save_render(full,w,out/f'{label}-reconstruction.jpg',renderer=fitted_renderer,source=path)
        rows.append({'role':label,'source':path,'source_sha256':morph.sha256_file(path),'faces':len(boxes),'source_to_target_rmse':face.rmse(sample,want),'output_to_target_rmse':face.rmse(estimate,want),'dimensions':[full.shape[1],full.shape[0]]});print(label,'rendered',flush=True)
    morph.dump({'scope':'Synthetic local-light target and reconstruction use the same face detections, but different mask shapes; detections unverified and masks extend into surroundings. One scene, two photos; not a real edit or professional quality validation. Fit at reduced resolution; JPEGs rendered at source dimensions.','size':a.size,'strength':a.strength,'settings':settings,'weights':w.tolist(),'photos':rows},out/'report.json')
if __name__=='__main__':main()
