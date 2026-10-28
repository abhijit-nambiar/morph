#!/usr/bin/env python3
"""Compare solvers on a saved randomized same-renderer experiment."""
import argparse,json,time
from pathlib import Path
import numpy as np
import morph,solver

def main():
    p=argparse.ArgumentParser();p.add_argument('cases');p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,default=45);p.add_argument('--size',type=int,default=256);a=p.parse_args(); data=json.loads(Path(a.cases).read_text());rows=[]
    for c in data['cases']:
        x=morph.load(c['representative'],a.size);true=morph.vec(c['expected']);target=morph.render(x,true);t=time.monotonic();pred,m=solver.fit_gauss_newton(x,target,iterations=a.iterations);elapsed=time.monotonic()-t
        held=morph.load(c['heldout_photo'],a.size);held_rmse=float(np.sqrt(np.mean((morph.render(held,pred)-morph.render(held,true))**2)))
        rows.append({'scene':c['scene'],'case':c['case'],'true':c['expected'],'pattern':c['fitted'],'gauss_newton':morph.recipe(pred),'pattern_fit_seconds':c['fit_seconds'],'gauss_newton_fit_seconds':elapsed,'pattern_heldout_rmse':c['heldout_rmse'],'gauss_newton_heldout_rmse':held_rmse,'gauss_newton_metrics':m})
        print(c['scene'],c['case'],'heldout',round(c['heldout_rmse'],5),'->',round(held_rmse,5),flush=True)
    summary={};
    for method in ('pattern','gauss_newton'):
        errors={k:float(np.mean([abs(r[method][k]-r['true'][k]) for r in rows])) for k in morph.NAMES};summary[method]={'mean_absolute_parameter_error':errors,'mean_heldout_rmse':float(np.mean([r[method+'_heldout_rmse'] for r in rows])),'median_seconds':float(np.median([r[method+'_fit_seconds'] for r in rows]))}
    morph.dump({'scope':'Comparison on same-renderer synthetic edits, same album and recipes; fixed 6000-pixel sample for Gauss-Newton versus up to 12000 for pattern search. Timing is local and approximate.','summary':summary,'cases':rows},a.out)
if __name__=='__main__':main()
