#!/usr/bin/env python3
"""Compare one vs multiple representative fits on the same randomized looks."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,gamut_renderer

def choose_anchors(group,heldout,max_anchors,size):
    rep=group['representative'];eligible=[p for p in group['photos'] if p not in heldout and p!=rep];candidates=[rep]+eligible
    ims={p:morph.load(p,size) for p in candidates}; features=np.array([morph.features(ims[p]) for p in candidates]);scale=np.maximum(np.std(features,axis=0),.04);z=(features-features.mean(0))/scale;selected=[0]
    while len(selected)<min(max_anchors,len(candidates)):
        dist=np.min(np.stack([np.sum((z-z[i])**2,axis=1) for i in selected]),axis=0);dist[selected]=-1;selected.append(int(np.argmax(dist)))
    return [(candidates[i],ims[candidates[i]]) for i in selected]

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('benchmark');p.add_argument('--out',required=True);p.add_argument('--anchors',type=int,default=3);p.add_argument('--size',type=int,default=256);a=p.parse_args()
    if a.anchors<1:p.error('--anchors must be positive')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));previous=json.loads(Path(a.benchmark).read_text());gmap={g['id']:g for g in groups};rows=[];cache={}
    for case in previous['cases']:
        g=gmap[case['scene']];held=[r['photo'] for r in case['heldout']['gamut']];key=(g['id'],tuple(held))
        if key not in cache:cache[key]=choose_anchors(g,set(held),a.anchors,a.size)
        anchors=cache[key];xs=[];ys=[];rng=np.random.default_rng(11)
        for path,im in anchors:
            target=morph.pillow_soft_target(im,'neutral',case['target_settings']);x=im.reshape(-1,3);y=target.reshape(-1,3);idx=rng.choice(len(x),min(3000,len(x)),replace=False);xs.append(x[idx]);ys.append(y[idx])
        fitted,metrics=gamut_renderer.fit(np.concatenate(xs),np.concatenate(ys)); outcomes=[]
        for item in case['heldout']['gamut']:
            image=morph.load(item['photo'],a.size);target=morph.pillow_soft_target(image,'neutral',case['target_settings']);pred=gamut_renderer.render(image,fitted);outcomes.append({'photo':item['photo'],'single_rmse':item['rmse'],'joint_rmse':float(np.sqrt(np.mean((target-pred)**2)))})
        rows.append({'scene':g['id'],'case':case['case'],'anchors':[p for p,_ in anchors],'recipe':gamut_renderer.recipe(fitted),'metrics':metrics,'heldout':outcomes});print(g['id'],case['case'],len(anchors),'anchors',flush=True)
    held=[x for c in rows for x in c['heldout']]
    if not held:raise ValueError('Benchmark contains no held-out photos')
    s=np.array([x['single_rmse'] for x in held]);j=np.array([x['joint_rmse'] for x in held]);morph.dump({'scope':'Synthetic targets from the same independent Pillow generator; additional anchors exclude held-out photos. Anchor selection and evaluation use the same unvalidated source album; no aesthetic review.','summary':{'cases':len(rows),'heldout':len(held),'single_mean_rmse':float(np.mean(s)),'joint_mean_rmse':float(np.mean(j)),'single_max_rmse':float(np.max(s)),'joint_max_rmse':float(np.max(j)),'joint_better':int(np.sum(j<s))},'cases':rows},a.out)
if __name__=='__main__':main()
