#!/usr/bin/env python3
"""Research-only: compare smooth grid lighting with simpler transferable mappings."""
import argparse
import json
from pathlib import Path
import numpy as np
import morph
import poly_color
import spatial_light
import offcenter_light_experiment as target_engine


def features(x):
    h,w=x.shape[:2]
    yy=(np.arange(h,dtype=np.float32)+.5)/h
    xx=(np.arange(w,dtype=np.float32)+.5)/w
    local=[]
    for cy in (.15,.5,.85):
        for cx in (.15,.5,.85):
            m=np.exp(-.5*(((xx[None,:]-cx)/.27)**2+((yy[:,None]-cy)/.27)**2))
            local.append(m[...,None]*x*(1-x))
    return np.concatenate([poly_color.features(x),*local],axis=-1)


def fit(x,y,penalty=.0001):
    rng=np.random.default_rng(0); idx=rng.choice(x.shape[0]*x.shape[1], min(x.shape[0]*x.shape[1],30000),replace=False)
    a=features(x).reshape(-1,37)[idx].astype(np.float64)
    b=(y-x).reshape(-1,3)[idx].astype(np.float64)
    return np.linalg.lstsq(a.T@a/len(a)+penalty*np.eye(37),a.T@b/len(a),rcond=None)[0]


def render(x,w): return np.clip(x+features(x)@w,0,1)
def rmse(x,y): return float(np.sqrt(np.mean((x-y)**2)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[]
    for g in groups:
        held=[p for p in g['photos'] if p!=g['representative']][:2]
        if not held: continue
        rep=morph.load(g['representative'],a.size); photos=[(str(p),morph.load(p,a.size)) for p in held]
        # Include no-local baseline to expose overfitting; target masks differ from fitting basis.
        for center in ((.2,.3),(.8,.7)):
            for strength in (-.3,0.,.3):
                y=target_engine.target(rep,settings,center,strength)
                wg,_=poly_color.fit(rep,y,regularization=.0001);wc=spatial_light.fit(rep,y,regularization=.0001);wf=fit(rep,y)
                for path,x in photos:
                    y=target_engine.target(x,settings,center,strength)
                    rows.append({'scene':g['id'],'photo':path,'center':list(center),'strength':strength,
                                 'global_rmse':rmse(poly_color.render(x,wg),y),'centered_rmse':rmse(spatial_light.render(x,wc),y),'grid_rmse':rmse(render(x,wf),y)})
        print(g['id'],len(held),flush=True)
    if not rows: raise ValueError('No held-out photos')
    summary={}
    for strength in (-.3,0.,.3):
        subset=[r for r in rows if r['strength']==strength]
        g=np.array([r['global_rmse'] for r in subset]);c=np.array([r['centered_rmse'] for r in subset]);f=np.array([r['grid_rmse'] for r in subset])
        summary[str(strength)]={'cases':len(subset),'grid_lower_than_global':int(np.sum(f<g)),'grid_lower_than_centered':int(np.sum(f<c)), 'global_mean_rmse':float(g.mean()),'centered_mean_rmse':float(c.mean()),'grid_mean_rmse':float(f.mean()),'grid_max_rmse':float(f.max())}
    morph.dump({'scope':'Exploratory synthetic test on one album; same frame-relative off-center target on representative and held-out photos, not semantic subject tracking. Grid and Gaussian target use different spatial basis functions. Same sampling and regularization across methods. No visual quality, unseen album or human-edit validation.', 'fit_size':a.size,'regularization':.0001,'summary':summary,'cases':rows},a.out)
if __name__=='__main__':main()
