#!/usr/bin/env python3
"""Compare scene transfer with and without an input-brightness heuristic."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color,gamut_renderer


def brightness(x):
    return float(np.median(x @ np.array([.2126,.7152,.0722],dtype=np.float32)))


def adjustment(reference,photo):
    # Approximate shift in the renderer's logit exposure units; no ground truth is used.
    ref=np.clip(reference,.02,.98);val=np.clip(photo,.02,.98)
    return float(np.clip((np.log(ref/(1-ref))-np.log(val/(1-val)))/np.log(2),-.5,.5))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sensitivity');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args()
    baseline=json.loads(Path(a.sensitivity).read_text());size=baseline['size'];reg=baseline['regularization'];inputs=[a.sensitivity];results=[];scene_rows=[]
    for scene in baseline['scenes']:
        names=[r['photo'] for r in scene['heldout']];pairs=[]
        for name in names:
            source=Path(name);edit=Path(a.edits)/source.name;inputs.extend([source,edit]);x=morph.load(source,size);y=morph.load(edit,size)
            if x.shape!=y.shape:raise ValueError('Different dimensions: '+name)
            pairs.append((str(source.resolve()),x,y,brightness(x)))
        weights=[poly_color.fit(x,y,regularization=reg)[0] for _,x,y,_ in pairs];rows=[]
        for j,(photo,x,y,light) in enumerate(pairs):
            values=[]
            for i,(rep,_,_,rep_light) in enumerate(pairs):
                if i==j:continue
                exposure=adjustment(rep_light,light);corrected=gamut_renderer.render(x,gamut_renderer.vec({'exposure':exposure}))
                standard=float(np.sqrt(np.mean((poly_color.render(x,weights[i])-y)**2)))
                normalized=float(np.sqrt(np.mean((poly_color.render(corrected,weights[i])-y)**2)))
                values.append({'representative':rep,'input_exposure_adjustment':exposure,'baseline_rmse':standard,'normalized_rmse':normalized})
            base=[v['baseline_rmse'] for v in values];norm=[v['normalized_rmse'] for v in values]
            row={'scene':scene['scene'],'photo':photo,'baseline_range':max(base)-min(base),'normalized_range':max(norm)-min(norm),'comparisons':values};rows.append(row);results.append(row)
        scene_rows.append({'scene':scene['scene'],'photos':len(rows),'baseline_mean_range':float(np.mean([r['baseline_range'] for r in rows])),'normalized_mean_range':float(np.mean([r['normalized_range'] for r in rows])),'photos_detail':rows});print(scene['scene'],len(rows),'photos',flush=True)
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False)
    comparisons=[v for r in results for v in r['comparisons']];base=np.array([v['baseline_rmse'] for v in comparisons]);norm=np.array([v['normalized_rmse'] for v in comparisons]);ranges=np.array([r['baseline_range'] for r in results]);new_ranges=np.array([r['normalized_range'] for r in results])
    morph.dump({'scope':'Exploratory paired test on the same selected photos and representative fits as the sensitivity study. For each held-out photo, estimate a bounded adjustment using original image median luminance relative to the representative original, then apply it before the representative mapping. No held-out edit informs this adjustment. Reduced-resolution pixel RMSE is not quality; same dimensions do not guarantee alignment, scene assignments and edited intent are unverified. Comparisons share photos and are not statistically independent. Do not use as an automatic normalizer without further validation.','size':size,'regularization':reg,'adjustment_limit':.5,'summary':{'scenes':len(scene_rows),'heldout_photos':len(results),'representative_photo_comparisons':len(comparisons),'baseline_mean_rmse':float(base.mean()),'normalized_mean_rmse':float(norm.mean()),'improved_comparisons':int(np.sum(norm<base-1e-8)),'worsened_comparisons':int(np.sum(norm>base+1e-8)),'unchanged_comparisons':int(np.sum(np.abs(norm-base)<=1e-8)),'baseline_mean_representative_range':float(ranges.mean()),'normalized_mean_representative_range':float(new_ranges.mean()),'lower_range_photos':int(np.sum(new_ranges<ranges))},'scenes':scene_rows},a.out)
if __name__=='__main__':main()
