#!/usr/bin/env python3
"""Compare global and spatial deterministic mappings on radial-light targets."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color,spatial_light

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--heldout',type=int,default=2);a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[];cases=[]
    for g in groups:
        original=morph.load(g['representative'],a.size);held=[p for p in g['photos'] if p!=g['representative']][:a.heldout]
        for strength in (-.3,0.,.3):
            target=spatial_light.target(original,settings,strength);global_fit,_=poly_color.fit(original,target,regularization=.0001);spatial_fit=spatial_light.fit(original,target,regularization=.0001);outcomes=[]
            for path in held:
                x=morph.load(path,a.size);want=spatial_light.target(x,settings,strength);gout=poly_color.render(x,global_fit);sout=spatial_light.render(x,spatial_fit);r={'photo':path,'strength':strength,'global_rmse':float(np.sqrt(np.mean((gout-want)**2))),'spatial_rmse':float(np.sqrt(np.mean((sout-want)**2)))};rows.append(r);outcomes.append(r)
            cases.append({'scene':g['id'],'representative':g['representative'],'strength':strength,'global_mapping':poly_color.serialize(global_fit),'spatial_weights':spatial_fit.tolist(),'heldout':outcomes})
        print(g['id'],len(held),'heldout',flush=True)
    summary={}
    for strength in (-.3,0.,.3):
        subset=[v for v in rows if v['strength']==strength];x=np.array([v['global_rmse'] for v in subset]);y=np.array([v['spatial_rmse'] for v in subset]);summary[str(strength)]={'heldout':len(subset),'spatial_better':int(np.sum(y<x)),'global_mean_rmse':float(x.mean()),'spatial_mean_rmse':float(y.mean()),'global_max_rmse':float(x.max()),'spatial_max_rmse':float(y.max())}
    morph.dump({'scope':'Synthetic target applies the same fixed radial light mask to each photo; this does not test whether such a mask follows the same subject across compositions. No geometry changes. The spatial model is experimental and not yet used for full-resolution album export.','settings':settings,'summary':summary,'cases':cases},a.out)
if __name__=='__main__':main()
