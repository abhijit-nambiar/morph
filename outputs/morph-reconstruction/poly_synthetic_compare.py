#!/usr/bin/env python3
"""Evaluate polynomial mapping against bounded fit on random independent targets."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color

def main():
    p=argparse.ArgumentParser();p.add_argument('benchmark');p.add_argument('--out',required=True);p.add_argument('--regularization',type=float,default=.001);p.add_argument('--size',type=int,default=256);a=p.parse_args();data=json.loads(Path(a.benchmark).read_text());cases=[];allrows=[]
    for c in data['cases']:
        x=morph.load(c['representative'],a.size);y=morph.pillow_soft_target(x,'neutral',c['target_settings']);w,metrics=poly_color.fit(x,y,regularization=a.regularization);rows=[]
        for row in c['heldout']['gamut']:
            x=morph.load(row['photo'],a.size);y=morph.pillow_soft_target(x,'neutral',c['target_settings']);err=float(np.sqrt(np.mean((poly_color.render(x,w)-y)**2)));entry={'photo':row['photo'],'bounded_rmse':row['rmse'],'polynomial_rmse':err,'polynomial_clipping':morph.clipping_diagnostic(x,w,renderer=poly_color.render)};rows.append(entry);allrows.append(entry)
        cases.append({'scene':c['scene'],'case':c['case'],'mapping':poly_color.serialize(w),'metrics':metrics,'heldout':rows})
    b=np.array([r['bounded_rmse'] for r in allrows]);v=np.array([r['polynomial_rmse'] for r in allrows]);morph.dump({'scope':'Independent Pillow soft targets; held-out photos from same album. Different models may fit different samples; near-limit flags are heuristics.','summary':{'heldout':len(allrows),'poly_lower_error':int(np.sum(v<b)),'bounded_mean_rmse':float(b.mean()),'poly_mean_rmse':float(v.mean()),'poly_flags':sum(r['polynomial_clipping']['flag'] for r in allrows)},'cases':cases},a.out)
if __name__=='__main__':main()
