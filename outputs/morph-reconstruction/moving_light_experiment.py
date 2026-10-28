#!/usr/bin/env python3
"""Evaluate transfer when the target lighting region moves between photographs."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color,grid_light
import offcenter_light_experiment as target_engine


def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[]
    shifts=((0.,0.),(.1,0.),(.2,0.),(.3,0.))
    for group in groups:
        held=[p for p in group['photos'] if p!=group['representative']][:2]
        if not held:continue
        rep=morph.load(group['representative'],a.size);photos=[(str(p),morph.load(p,a.size)) for p in held]
        for strength in (-.3,.3):
            original_center=(.3,.4);ref=target_engine.target(rep,settings,original_center,strength)
            wg,_=poly_color.fit(rep,ref,regularization=.0001);wf,_=grid_light.fit_with_metrics(rep,ref,regularization=.0001)
            for dx,dy in shifts:
                center=(original_center[0]+dx,original_center[1]+dy)
                for path,x in photos:
                    target=target_engine.target(x,settings,center,strength)
                    rows.append({'scene':group['id'],'photo':path,'strength':strength,'shift':[dx,dy],
                        'source_rmse':rmse(x,target),'global_rmse':rmse(poly_color.render(x,wg),target),
                        'grid_rmse':rmse(grid_light.render(x,wf),target)})
        print(group['id'],len(held),flush=True)
    if not rows:raise ValueError('No held-out photos')
    summary={}
    for dx,dy in shifts:
        subset=[r for r in rows if r['shift']==[dx,dy]];g=np.array([r['global_rmse'] for r in subset]);f=np.array([r['grid_rmse'] for r in subset]);source=np.array([r['source_rmse'] for r in subset])
        summary[str(dx)]={'cases':len(subset),'grid_lower_than_global':int(np.sum(f<g)),'grid_lower_than_source':int(np.sum(f<source)), 'global_mean_rmse':float(g.mean()),'grid_mean_rmse':float(f.mean()),'source_mean_rmse':float(source.mean())}
    morph.dump({'scope':'Synthetic independent Gaussian target with horizontally shifted mask in held-out photos; movement is prescribed, not detected. One album, unverified proposed scenes, same fitting pixels and penalty. The target shape is similar to the grid basis. RMSE is not photographic quality; this is not a real subject-tracking test.', 'size':a.size,'regularization':.0001,'settings':settings,'summary':summary,'cases':rows},a.out)
if __name__=='__main__':main()
