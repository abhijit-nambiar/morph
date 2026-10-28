#!/usr/bin/env python3
"""Compare a selective-color renderer against global controls on real pairs."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,selective

def main():
    p=argparse.ArgumentParser();p.add_argument('transfer_report');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--regularization',type=float,default=0);p.add_argument('--renderer',choices=['selective','gamut'],default='selective');a=p.parse_args();old=json.loads(Path(a.transfer_report).read_text()); scenes=[];all_rows=[]
    if a.renderer=='gamut':
        if a.regularization:raise ValueError('gamut experiment does not support regularization')
        import gamut_renderer as model
    else:model=selective
    for g in old['scenes']:
        rep=Path(g['fitting_representative']);tgt=None
        for r in g['heldout_results']:
            tgt=Path(r['edit']).parent/rep.name;break
        if tgt is None:scenes.append({'id':g['id'],'status':'no heldout pairs'});continue
        x=morph.load(rep,a.size);y=morph.load(tgt,a.size);fit,metrics=model.fit(x,y,regularization=a.regularization) if a.renderer=='selective' else model.fit(x,y);rows=[]
        for r in g['heldout_results']:
            x=morph.load(r['photo'],a.size);y=morph.load(r['edit'],a.size);pred=model.render(x,fit);err=float(np.sqrt(np.mean((pred-y)**2)))
            row={'photo':r['photo'],'identity_rmse':r['identity_rmse'],'global_rmse':r['transfer_rmse'],'selective_rmse':err};rows.append(row);all_rows.append(row)
        scenes.append({'id':g['id'],'representative':str(rep),'fit_metrics':metrics,'recipe':model.recipe(fit),'heldout':rows});print(g['id'],len(rows),flush=True)
    global_errors=np.array([r['global_rmse'] for r in all_rows]);selective_errors=np.array([r['selective_rmse'] for r in all_rows]);
    morph.dump({'scope':'Exploratory comparison on one album using unvalidated chronological groups and same-sized pairs; fitting and evaluation use different photos, but no unseen album. Selective colors are RGB heuristics, not semantic subject masks.','renderer':a.renderer,'regularization':a.regularization,'summary':{'heldout_pairs':len(all_rows),'candidate_better':int(np.sum(selective_errors<global_errors)),'global_median_rmse':float(np.median(global_errors)),'candidate_median_rmse':float(np.median(selective_errors)),'global_mean_rmse':float(np.mean(global_errors)),'candidate_mean_rmse':float(np.mean(selective_errors))},'scenes':scenes},a.out)
if __name__=='__main__':main()
