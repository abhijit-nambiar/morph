#!/usr/bin/env python3
"""Compare experimental grid illumination with earlier real-pair fits."""
import argparse,json
from pathlib import Path
import numpy as np
import morph, poly_color, spatial_light, grid_light_experiment

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args();report=json.loads(Path(a.report).read_text());rows=[]
    size=report['size']; penalty=report['regularization']
    for s in report['scenes']:
        rep=Path(s['representative']);x=morph.load(rep,size);y=morph.load(Path(a.edits)/rep.name,size);w=grid_light_experiment.fit(x,y,penalty)
        for row in s['heldout']:
            photo=Path(row['photo']);x=morph.load(photo,size);y=morph.load(Path(a.edits)/photo.name,size);raw=x+grid_light_experiment.features(x)@w
            rows.append({'scene':s['id'],'photo':str(photo),'global_rmse':row['polynomial_rmse'],'centered_rmse':row['spatial_rmse'],'grid_rmse':grid_light_experiment.rmse(np.clip(raw,0,1),y),'grid_out_of_gamut_fraction':float(np.mean(np.any((raw<0)|(raw>1),axis=-1)))})
        print(s['id'],len(s['heldout']),flush=True)
    if not rows:raise ValueError('No held-out photos')
    g=np.array([r['global_rmse'] for r in rows]);c=np.array([r['centered_rmse'] for r in rows]);f=np.array([r['grid_rmse'] for r in rows])
    morph.dump({'scope':'Exploratory real-pair pixel RMSE in one album with unverified groups and pair alignment; edits may have different intent. Same representative, sampled pixels and regularization. Does not assess visual quality or subject-aware edits.', 'summary':{'heldout':len(rows),'grid_lower_than_global':int(np.sum(f<g)),'grid_lower_than_centered':int(np.sum(f<c)),'global_mean_rmse':float(g.mean()),'centered_mean_rmse':float(c.mean()),'grid_mean_rmse':float(f.mean()),'grid_median_rmse':float(np.median(f)),'grid_mean_out_of_gamut_fraction':float(np.mean([r['grid_out_of_gamut_fraction'] for r in rows]))},'cases':rows},a.out)
if __name__=='__main__':main()
