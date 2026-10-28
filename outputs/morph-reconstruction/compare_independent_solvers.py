#!/usr/bin/env python3
"""Compare solver transfer errors on separately generated Pillow targets."""
import argparse,json,time
from pathlib import Path
import numpy as np
import morph,solver

def main():
    p=argparse.ArgumentParser();p.add_argument('benchmark');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=320);a=p.parse_args();data=json.loads(Path(a.benchmark).read_text());rows=[]
    for c in data['cases']:
        x=morph.load(c['representative'],a.size);y=morph.pillow_target(x,c['look']); t=time.monotonic();fit,metrics=solver.fit_gauss_newton(x,y);elapsed=time.monotonic()-t;held=[]
        for r in c['heldout_photo_results']:
            im=morph.load(r['photo'],a.size);truth=morph.pillow_target(im,c['look']);held.append({'photo':r['photo'],'pattern_rmse':r['rmse'],'gauss_newton_rmse':float(np.sqrt(np.mean((morph.render(im,fit)-truth)**2)))})
        rows.append({'scene':c['scene'],'look':c['look'],'fitted_recipe':morph.recipe(fit),'metrics':metrics,'fit_seconds':elapsed,'heldout_results':held});print(c['scene'],c['look'],flush=True)
    held=[r for c in rows for r in c['heldout_results']];morph.dump({'scope':'Independent Pillow-generated targets on the same album; methods use different randomly sampled pixel subsets, so this is not a controlled runtime benchmark.','summary':{'cases':len(rows),'heldout':len(held),'pattern_mean_rmse':float(np.mean([r['pattern_rmse'] for r in held])),'gauss_newton_mean_rmse':float(np.mean([r['gauss_newton_rmse'] for r in held])),'gauss_newton_improved':sum(r['gauss_newton_rmse']<r['pattern_rmse'] for r in held)},'cases':rows},a.out)
if __name__=='__main__':main()
