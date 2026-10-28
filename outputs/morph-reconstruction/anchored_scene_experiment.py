#!/usr/bin/env python3
"""Compare scene fits constrained toward a jointly fitted album recipe."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,gamut_renderer,solver

def main():
    p=argparse.ArgumentParser();p.add_argument('scene_report');p.add_argument('album_report');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args();data=json.loads(Path(a.scene_report).read_text());baseline=json.loads(Path(a.album_report).read_text());shared=gamut_renderer.vec(baseline['album_recipe']);factors=(.00001,.0001,.001);scenes=[];rows=[]
    for scene in data['scenes']:
        if 'recipe' not in scene or not scene['heldout']:continue
        rep=scene['representative'];x=morph.load(rep,a.size);y=morph.load(Path(a.edits)/Path(rep).name,a.size);fits={};recipes={}
        for f in factors:
            q,metrics=solver.fit_gauss_newton(x,y,renderer=gamut_renderer.render,bounds=gamut_renderer.BOUNDS,penalty=np.full(len(shared),f),prior=shared);recipes[f]=q;fits[str(f)]={'recipe':gamut_renderer.recipe(q),'metrics':metrics,'distance_from_shared':float(np.linalg.norm(q-shared))}
        held=[]
        for item in scene['heldout']:
            img=morph.load(item['photo'],a.size);target=morph.load(Path(a.edits)/Path(item['photo']).name,a.size);r={'scene':scene['id'],'photo':item['photo'],'scene_rmse':item['selective_rmse'],'album_rmse':float(np.sqrt(np.mean((gamut_renderer.render(img,shared)-target)**2))),'anchored':{}}
            for f,q in recipes.items():r['anchored'][str(f)]=float(np.sqrt(np.mean((gamut_renderer.render(img,q)-target)**2)))
            held.append(r);rows.append(r)
        scenes.append({'scene':scene['id'],'representative':rep,'fits':fits,'heldout':held});print(scene['id'],len(held),flush=True)
    summary={}
    for f in factors:
        a_=np.array([r['anchored'][str(f)] for r in rows]);s=np.array([r['scene_rmse'] for r in rows]);b=np.array([r['album_rmse'] for r in rows]);summary[str(f)]={'heldout':len(rows),'mean_rmse':float(a_.mean()),'median_rmse':float(np.median(a_)),'better_than_scene':int(np.sum(a_<s-1e-6)),'worse_than_scene':int(np.sum(a_>s+1e-6)),'better_than_album':int(np.sum(a_<b-1e-6))}
    morph.dump({'scope':'Three penalty strengths tested on the same album and held-out photos. Scene fits are regularized toward an album fit from the same representatives; pixel error does not measure aesthetic album consistency. Tuning and grouping are not validated on an unseen album.','shared_recipe':baseline['album_recipe'],'summary':summary,'scenes':scenes},a.out)
if __name__=='__main__':main()
