#!/usr/bin/env python3
"""Measure recovery of randomized known renderer recipes and held-out transfer."""
import argparse,json,time
from pathlib import Path
import numpy as np
import morph

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--recipes',type=int,default=6);p.add_argument('--seed',type=int,default=53);p.add_argument('--size',type=int,default=256);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));rng=np.random.default_rng(a.seed);limits=np.array([.3,.12,.1,.2,.2,.2,.2]);cases=[]
    for g in groups:
        image=morph.load(g['representative'],a.size); held=next((x for x in g['photos'] if x!=g['representative']),None);other=morph.load(held,a.size) if held else None
        for i in range(a.recipes):
            expected=rng.uniform(-limits,limits);target=morph.render(image,expected); t=time.monotonic();pred,info=morph.fit(image,target);elapsed=time.monotonic()-t
            pred_img=morph.render(image,pred); rep_rmse=float(np.sqrt(np.mean((pred_img-target)**2))); transfer_rmse=float(np.sqrt(np.mean((morph.render(other,pred)-morph.render(other,expected))**2))) if other is not None else None
            cases.append({'scene':g['id'],'case':i,'representative':g['representative'],'heldout_photo':held,'expected':morph.recipe(expected),'fitted':morph.recipe(pred),'representative_rmse':rep_rmse,'heldout_rmse':transfer_rmse,'fit_seconds':elapsed,'metrics':info})
        print(g['id'],a.recipes,'recipes',flush=True)
    per_control={};
    for name in morph.NAMES:
        e=np.array([abs(c['expected'][name]-c['fitted'][name]) for c in cases]);per_control[name]={'mean_absolute_error':float(e.mean()),'max_absolute_error':float(e.max())}
    held=[r['heldout_rmse'] for r in cases if r['heldout_rmse'] is not None]
    morph.dump({'scope':'Targets and fitter share a renderer. Random recipes are sampled within moderate preset bounds; there is no JPEG generation or held-out album. Slider parameters can be non-identifiable even when pixels match.','seed':a.seed,'sampled_abs_limits':dict(zip(morph.NAMES,limits.tolist())),'summary':{'cases':len(cases),'per_control':per_control,'mean_representative_rmse':float(np.mean([c['representative_rmse'] for c in cases])),'mean_heldout_photo_rmse':float(np.mean(held)) if held else None,'max_heldout_photo_rmse':float(np.max(held)) if held else None},'cases':cases},a.out)
if __name__=='__main__':main()
