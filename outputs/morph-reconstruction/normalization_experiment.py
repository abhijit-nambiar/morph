#!/usr/bin/env python3
"""Evaluate a composition-sensitive exposure-normalization heuristic, offline."""
import argparse,json,math
from pathlib import Path
import numpy as np
import morph

def ymedian(x):
    lum=x@np.array([.2126,.7152,.0722],dtype=np.float32);return float(np.median(lum))
def main():
    p=argparse.ArgumentParser();p.add_argument('transfer_report');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args(); data=json.loads(Path(a.transfer_report).read_text());rows=[]
    for scene in data['scenes']:
        ref=morph.load(scene['fitting_representative'],a.size);base=ymedian(ref);recipe=morph.vec(scene['fitted_recipe'])
        for r in scene['heldout_results']:
            x=morph.load(r['photo'],a.size); target=morph.load(r['edit'],a.size); median=ymedian(x); correction=float(np.clip(math.log2(max(base,.01)/max(median,.01)),-.35,.35)); adjusted=np.clip(x*(2**correction),0,1)
            pred=morph.render(adjusted,recipe); err=float(np.sqrt(np.mean((pred-target)**2)));rows.append({'scene':scene['id'],'photo':r['photo'],'correction_stops':correction,'identity_rmse':r['identity_rmse'],'transfer_rmse':r['transfer_rmse'],'normalized_rmse':err})
        print(scene['id'],'done',flush=True)
    if rows:
        fixed=np.array([x['transfer_rmse'] for x in rows]); adjusted=np.array([x['normalized_rmse'] for x in rows]);ident=np.array([x['identity_rmse'] for x in rows]); summary={'pairs':len(rows),'improved_over_fixed':int(np.sum(adjusted<fixed)),'improved_over_original':int(np.sum(adjusted<ident)),'median_fixed_rmse':float(np.median(fixed)),'median_normalized_rmse':float(np.median(adjusted)),'median_original_rmse':float(np.median(ident))}
    else:summary={'pairs':0}
    morph.dump({'method':'Rescale each source toward its fitting representative median luminance, clamped to +/-0.35 stops, before the fitted recipe. Same-size pairs only; registration and proposed groups are unvalidated.','summary':summary,'rows':rows},a.out)
if __name__=='__main__':main()
