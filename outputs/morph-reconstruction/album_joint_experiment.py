#!/usr/bin/env python3
"""Compare shared-album and per-scene fits for one deterministic target look."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--heldout',type=int,default=3);a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rng=np.random.default_rng(123);anchors=[];scene_fits={}
    for g in groups:
        image=morph.load(g['representative'],a.size);target=morph.pillow_soft_target(image,'neutral',settings);p_,metrics=gamut_renderer.fit(image,target);scene_fits[g['id']]={'recipe':gamut_renderer.recipe(p_),'metrics':metrics};x=image.reshape(-1,3);y=target.reshape(-1,3);idx=rng.choice(len(x),min(3000,len(x)),replace=False);anchors.append((x[idx],y[idx]))
    joint,metrics=gamut_renderer.fit(np.concatenate([a[0] for a in anchors]),np.concatenate([a[1] for a in anchors]));rows=[]
    for g in groups:
        individual=gamut_renderer.vec(scene_fits[g['id']]['recipe']);photos=[p for p in g['photos'] if p!=g['representative']][:a.heldout]
        for path in photos:
            x=morph.load(path,a.size);y=morph.pillow_soft_target(x,'neutral',settings);error=lambda p:float(np.sqrt(np.mean((gamut_renderer.render(x,p)-y)**2)));rows.append({'scene':g['id'],'photo':path,'scene_rmse':error(individual),'album_rmse':error(joint)})
        print(g['id'],len(photos),'heldout',flush=True)
    s=np.array([r['scene_rmse'] for r in rows]);t=np.array([r['album_rmse'] for r in rows]);morph.dump({'scope':'Same deterministic independent target settings applied to all scenes; same album. Heldout photos excluded from fitting but scene representatives were used. Pixel error does not establish visual album consistency or professional quality.','settings':settings,'shared_recipe':gamut_renderer.recipe(joint),'shared_fit_metrics':metrics,'scene_fits':scene_fits,'summary':{'representatives':len(groups),'heldout':len(rows),'shared_lower_error':int(np.sum(t<s)),'scene_mean_rmse':float(s.mean()),'album_mean_rmse':float(t.mean()),'scene_max_rmse':float(s.max()),'album_max_rmse':float(t.max())},'heldout':rows},a.out)
if __name__=='__main__':main()
