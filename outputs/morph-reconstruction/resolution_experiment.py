#!/usr/bin/env python3
"""Check thumbnail fit sizes against independent full-resolution pixel samples."""
import argparse,json,time
from pathlib import Path
import numpy as np
import morph,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--groups-limit',type=int,default=4);p.add_argument('--sample',type=int,default=80000);a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()))[:a.groups_limit];settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[]
    for g in groups:
        full=morph.load(g['representative']);rng=np.random.default_rng(923);pix=full.reshape(-1,3);indices=rng.choice(len(pix),min(a.sample,len(pix)),replace=False);sample=pix[indices].reshape(-1,1,3);actual=morph.pillow_soft_target(sample,'neutral',settings).reshape(-1,3);del full,pix
        case={'scene':g['id'],'representative':g['representative'],'sampled_full_resolution_pixels':len(sample),'fits':[]}
        for size in (320,640,1024):
            x=morph.load(g['representative'],size);target=morph.pillow_soft_target(x,'neutral',settings);start=time.monotonic();recipe,metrics=gamut_renderer.fit(x,target);seconds=time.monotonic()-start;prediction=gamut_renderer.render(sample,recipe).reshape(-1,3);err=float(np.sqrt(np.mean((actual-prediction)**2)));case['fits'].append({'fit_max_dimension':size,'full_resolution_sample_rmse':err,'fit_seconds':seconds,'metrics':metrics,'recipe':gamut_renderer.recipe(recipe)})
        rows.append(case);print(g['id'],[(x['fit_max_dimension'],round(x['full_resolution_sample_rmse'],5)) for x in case['fits']],flush=True)
    summaries=[]
    for size in (320,640,1024):
        results=[x for c in rows for x in c['fits'] if x['fit_max_dimension']==size];summaries.append({'fit_max_dimension':size,'mean_full_resolution_sample_rmse':float(np.mean([x['full_resolution_sample_rmse'] for x in results])),'mean_fit_seconds':float(np.mean([x['fit_seconds'] for x in results]))})
    morph.dump({'scope':'Samples original-resolution representative pixels and independently generates the target from those same pixels; no JPEG encode, transfer photos, crop, or unseen album. Per-pixel target generation is valid for pillow-soft. Full-image resampling, clipping and visual quality are not established by RMSE.','target_settings':settings,'summary':summaries,'cases':rows},a.out)
if __name__=='__main__':main()
