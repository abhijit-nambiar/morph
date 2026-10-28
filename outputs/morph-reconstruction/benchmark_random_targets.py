#!/usr/bin/env python3
"""Random independent LUT/HSV targets, comparing scene transfer and clipping."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,selective,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--looks',type=int,default=4);p.add_argument('--seed',type=int,default=91);p.add_argument('--size',type=int,default=256);p.add_argument('--heldout',type=int,default=2);p.add_argument('--scale',type=float,default=1);a=p.parse_args()
    if not 0<a.scale<=2:p.error('--scale must be > 0 and <= 2')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));rng=np.random.default_rng(a.seed);limits=np.array([.24,.22,.22,.22,.12,.24,.24])*a.scale;records=[];visual=[];rejected=0
    for g in groups:
        x=morph.load(g['representative'],a.size);photos=[s for s in g['photos'] if s!=g['representative']][:a.heldout];held=[(p,morph.load(p,a.size)) for p in photos]
        for i in range(a.looks):
            for _ in range(100):
                settings=dict(zip(morph.SOFT_NAMES,rng.uniform(-limits,limits).tolist()))
                try:target=morph.pillow_soft_target(x,'neutral',settings);break
                except ValueError:rejected+=1
            else:raise ValueError('Unable to sample valid bounded tone settings')
            models={};result={}
            for name in ['global','selective','gamut']:
                if name=='global':fit,metrics=morph.run_fit(x,target,method='gauss-newton');renderer=morph.render;encode=morph.recipe
                else:
                    mod=selective if name=='selective' else gamut_renderer;fit,metrics=mod.fit(x,target);renderer=mod.render;encode=mod.recipe
                models[name]={'recipe':encode(fit),'fit_metrics':metrics};result[name]=[]
                for path,img in held:
                    truth=morph.pillow_soft_target(img,'neutral',settings);pred=renderer(img,fit);risk=morph.clipping_diagnostic(img,fit,renderer)
                    high=float(np.mean(np.any(truth>=254/255,axis=-1)));low=float(np.mean(np.all(truth<=1/255,axis=-1)))
                    target_flag=(high-risk['before']['high']>.01) or (low-risk['before']['low']>.01)
                    result[name].append({'photo':path,'rmse':float(np.sqrt(np.mean((truth-pred)**2))),'clipping':risk,'target_near_white':high,'target_near_black':low,'target_flag':target_flag})
            records.append({'scene':g['id'],'case':i,'representative':g['representative'],'target_settings':settings,'models':models,'heldout':result})
            if len(visual)<12:
                visual.append((g['id']+' '+str(i+1),(x,target,gamut_renderer.render(x,gamut_renderer.vec(models['gamut']['recipe'])))))
        print(g['id'],a.looks,'targets',len(held),'heldout photos',flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);s={}
    for name in ['global','selective','gamut']:
        rows=[r for c in records for r in c['heldout'][name]];v=[r['rmse'] for r in rows];s[name]={'heldout_combinations':len(rows),'mean_rmse':float(np.mean(v)) if v else None,'max_rmse':float(max(v)) if v else None,'clipping_flags':sum(r['clipping']['flag'] for r in rows),'target_clipping_flags':sum(r['target_flag'] for r in rows)}
    pages=morph.comparison_sheet(visual,out,labels='original / independent target / bounded reconstruction');morph.dump({'scope':'Randomized independent Pillow tone LUT + HSV hue-range targets. Multiple held-out photos per proposed group but the same source album; no independent album or aesthetic assessment. Nonmonotonic target curves are rejected before fitting.','seed':a.seed,'scale':a.scale,'rejected_targets':rejected,'abs_limits':dict(zip(morph.SOFT_NAMES,limits.tolist())),'summary':s,'cases':records,'comparison_sheets':pages},out/'report.json')
if __name__=='__main__':main()
