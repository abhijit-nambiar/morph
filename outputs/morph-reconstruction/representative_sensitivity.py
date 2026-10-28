#!/usr/bin/env python3
"""Measure how much the choice of fitted representative changes held-out reconstruction."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import morph, poly_color


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('splits',help='Prior held-out scene report defining eligible same-size pairs')
    parser.add_argument('edits');parser.add_argument('--out',required=True);parser.add_argument('--size',type=int,default=128);parser.add_argument('--regularization',type=float,default=.001)
    a=parser.parse_args()
    if not 64<=a.size<=512 or not np.isfinite(a.regularization) or a.regularization<0:raise ValueError('Invalid size or regularization')
    splits=json.loads(Path(a.splits).read_text());inputs=[a.splits];scenes=[];all_rows=[]
    for scene in splits['scenes']:
        originals=[scene['representative']]+[r['photo'] for r in scene['heldout']]
        if len(originals)<3:continue
        if len(set(originals))!=len(originals):raise ValueError('Duplicate scene photo')
        pairs=[]
        for name in originals:
            original=Path(name);edit=Path(a.edits)/original.name
            with Image.open(original) as im: original_size=ImageOps.exif_transpose(im).size
            with Image.open(edit) as im: edit_size=ImageOps.exif_transpose(im).size
            if original_size!=edit_size:raise ValueError('Dimension mismatch: '+str(original))
            inputs.extend([original,edit]);pairs.append((str(original.resolve()),morph.load(original,a.size),morph.load(edit,a.size)))
        weights=[poly_color.fit(x,y,regularization=a.regularization)[0] for _,x,y in pairs]
        rows=[]
        for j,(photo,x,y) in enumerate(pairs):
            values=[]
            for i,(rep,_,_) in enumerate(pairs):
                if i==j:continue
                error=float(np.sqrt(np.mean((poly_color.render(x,weights[i])-y)**2)))
                values.append({'representative':rep,'rmse':error})
            scores=[v['rmse'] for v in values]
            rows.append({'photo':photo,'source_rmse':float(np.sqrt(np.mean((x-y)**2))),'representatives':values,'rmse_min':min(scores),'rmse_max':max(scores),'rmse_range':max(scores)-min(scores)})
        all_rows.extend(rows);scenes.append({'scene':scene['id'],'pairs':len(pairs),'mean_rmse_range':float(np.mean([r['rmse_range'] for r in rows])),'heldout':rows});print(scene['id'],len(pairs),'held-out photos',flush=True)
    if not all_rows:raise ValueError('No scenes with at least three eligible pairs')
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False)
    spread=np.array([r['rmse_range'] for r in all_rows]);morph.dump({'scope':'Exploratory leave-one-out representative sensitivity on a single album and preselected same-size pairs. Each target photo is excluded from every fit used to predict it. Same dimensions do not establish alignment; scenes are unverified and edited photos may have different intentions. Ranges are descriptive hindsight, not an automatic selection method or a measure of photographic quality. Pixel RMSE is computed at reduced resolution.','size':a.size,'regularization':a.regularization,'summary':{'scenes':len(scenes),'heldout_photos':len(all_rows),'mean_rmse_range':float(np.mean(spread)),'median_rmse_range':float(np.median(spread)),'max_rmse_range':float(np.max(spread))},'scenes':scenes},a.out)
if __name__=='__main__':main()
