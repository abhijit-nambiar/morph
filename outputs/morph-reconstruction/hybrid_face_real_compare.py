#!/usr/bin/env python3
"""Research-only comparison of grid plus face features on held-out human edits."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,grid_light,face_light_experiment as face


def features(x,boxes):return np.concatenate([grid_light.features(x),face.mask(x.shape,boxes)[...,None]*x*(1-x)],axis=-1)
def fit(x,y,boxes,penalty):
    rng=np.random.default_rng(0);idx=rng.choice(x.shape[0]*x.shape[1],min(x.shape[0]*x.shape[1],30000),replace=False)
    f=features(x,boxes).reshape(-1,40)[idx].astype(np.float64);b=(y-x).reshape(-1,3)[idx].astype(np.float64)
    return np.linalg.lstsq(f.T@f/len(f)+penalty*np.eye(40),f.T@b/len(f),rcond=None)[0]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline');p.add_argument('detections');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args()
    baseline=json.loads(Path(a.baseline).read_text());lookup={str(Path(r['path']).resolve()):r for r in json.loads(Path(a.detections).read_text())};rows=[]
    for s in baseline['scenes']:
        rep=s['representative'];x=morph.load(rep,baseline['size']);y=morph.load(Path(a.edits)/Path(rep).name,baseline['size']);w=fit(x,y,lookup[rep]['faces'],baseline['regularization'])
        for r in s['heldout']:
            photo=r['photo'];x=morph.load(photo,baseline['size']);y=morph.load(Path(a.edits)/Path(photo).name,baseline['size']);raw=x+features(x,lookup[photo]['faces'])@w
            rows.append({'scene':s['scene'],'photo':photo,'grid_rmse':r['grid_rmse'],'face_rmse':r['face_rmse'],'hybrid_rmse':face.rmse(np.clip(raw,0,1),y),'hybrid_out_of_gamut_fraction':float(np.mean(np.any((raw<0)|(raw>1),axis=-1)))})
        print(s['scene'],len(s['heldout']),flush=True)
    if not rows:raise ValueError('No held-out photos')
    grid=np.array([r['grid_rmse'] for r in rows]);f=np.array([r['face_rmse'] for r in rows]);h=np.array([r['hybrid_rmse'] for r in rows])
    morph.dump({'scope':'One album, unverified scene grouping and face detections, same representatives, fitting samples and regularization as baseline. Real edit alignment and intent are unverified; pixel error does not assess photographic quality.','summary':{'heldout':len(rows),'hybrid_lower_than_grid':int(np.sum(h<grid)),'hybrid_lower_than_face':int(np.sum(h<f)),'grid_mean_rmse':float(grid.mean()),'face_mean_rmse':float(f.mean()),'hybrid_mean_rmse':float(h.mean()),'hybrid_median_rmse':float(np.median(h)),'hybrid_mean_out_of_gamut_fraction':float(np.mean([r['hybrid_out_of_gamut_fraction'] for r in rows]))},'rows':rows},a.out)
if __name__=='__main__':main()
