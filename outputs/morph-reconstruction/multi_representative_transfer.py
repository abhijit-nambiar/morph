#!/usr/bin/env python3
"""Leave-one-photo-out comparison of individual and pooled scene mappings."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import morph,poly_color


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('sensitivity');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args()
    prior=json.loads(Path(a.sensitivity).read_text());size=prior['size'];reg=prior['regularization'];inputs=[a.sensitivity];scenes=[];all_rows=[]
    for scene in prior['scenes']:
        paths=[r['photo'] for r in scene['heldout']]
        if len(paths)<3 or len(set(paths))!=len(paths):raise ValueError('Need at least three unique photos')
        pairs=[]
        for photo in paths:
            original=Path(photo);edited=Path(a.edits)/original.name;inputs.extend([original,edited])
            with Image.open(original) as im: original_size=ImageOps.exif_transpose(im).size
            with Image.open(edited) as im: edited_size=ImageOps.exif_transpose(im).size
            if original_size!=edited_size:raise ValueError('Different image dimensions: '+photo)
            x=morph.load(original,size);y=morph.load(edited,size);pairs.append((photo,x,y))
        mappings=[poly_color.fit(x,y,regularization=reg)[0] for _,x,y in pairs];rows=[]
        for j,(photo,x,y) in enumerate(pairs):
            training=[w for i,w in enumerate(mappings) if i!=j];individual=[float(np.sqrt(np.mean((poly_color.render(x,w)-y)**2))) for w in training]
            mean_weights=np.mean(training,axis=0);median_weights=np.median(training,axis=0)
            rmse=lambda z:float(np.sqrt(np.mean((z-y)**2)))
            row={'photo':photo,'training_pairs':len(training),'source_rmse':rmse(x),'configured_representative_rmse':None if j==0 else rmse(poly_color.render(x,mappings[0])),'single_representative_mean_rmse':float(np.mean(individual)),'single_representative_min_rmse':min(individual),'single_representative_max_rmse':max(individual),'mean_mapping_rmse':rmse(poly_color.render(x,mean_weights)),'median_mapping_rmse':rmse(poly_color.render(x,median_weights))};rows.append(row);all_rows.append(row)
        scenes.append({'scene':scene['scene'],'photos':len(rows),'heldout':rows});print(scene['scene'],len(rows),'held-out',flush=True)
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False)
    single=np.array([r['single_representative_mean_rmse'] for r in all_rows]);mean=np.array([r['mean_mapping_rmse'] for r in all_rows]);median=np.array([r['median_mapping_rmse'] for r in all_rows]);source=np.array([r['source_rmse'] for r in all_rows])
    fixed=[r for r in all_rows if r['configured_representative_rmse'] is not None]
    morph.dump({'scope':'One album, preselected same-size pairs and proposed unverified scenes. Each held-out photo and its edit are excluded from all training mappings; other edited examples in that scene are required. We average coefficients fitted separately to other pairs, not jointly fitted parameters. Mean individual-representative error gives each held-out photo equal weight and is not an oracle or an estimate for a specific choice of representative; averaging predictions generally reduces mean squared error by construction. The configured representative is the first photo in each original sensitivity scene; its own row is excluded from the fixed comparison. Equal size does not ensure alignment and human edits may differ intentionally. Reduced-resolution RGB RMSE is not visual quality.','size':size,'regularization':reg,'summary':{'scenes':len(scenes),'heldout_photos':len(all_rows),'source_mean_rmse':float(source.mean()),'single_representative_mean_rmse':float(single.mean()),'mean_mapping_mean_rmse':float(mean.mean()),'median_mapping_mean_rmse':float(median.mean()),'mean_mapping_lower_than_single_average':int(np.sum(mean<single)),'median_mapping_lower_than_single_average':int(np.sum(median<single)),'fixed_representative_comparison':{'photos':len(fixed),'fixed_mean_rmse':float(np.mean([r['configured_representative_rmse'] for r in fixed])),'pooled_mean_rmse':float(np.mean([r['mean_mapping_rmse'] for r in fixed])),'pooled_lower':sum(r['mean_mapping_rmse']<r['configured_representative_rmse'] for r in fixed)}},'scenes':scenes},a.out)
if __name__=='__main__':main()
