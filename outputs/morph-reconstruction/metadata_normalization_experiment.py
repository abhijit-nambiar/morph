#!/usr/bin/env python3
"""Explore camera-metadata-based exposure correction against real edited pairs."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image
import morph,gamut_renderer

def exposure_proxy(path):
    with Image.open(path) as im:
        d=im.getexif().get_ifd(34665);shutter=d.get(33434);aperture=d.get(33437);iso=d.get(34855)
    try:
        if isinstance(iso,(tuple,list)):iso=iso[0]
        shutter=float(shutter);aperture=float(aperture);iso=float(iso)
        if min(shutter,aperture,iso)<=0:return None
        return shutter*iso/(aperture*aperture)
    except (TypeError,ValueError,ZeroDivisionError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('comparison');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args();data=json.loads(Path(a.comparison).read_text());rows=[];skips=[]
    for scene in data['scenes']:
        if 'recipe' not in scene:continue
        ref=exposure_proxy(scene['representative']);look=gamut_renderer.vec(scene['recipe'])
        if not ref:skips.append({'scene':scene['id'],'reason':'missing representative metadata'});continue
        for item in scene['heldout']:
            photo=item['photo'];metadata=exposure_proxy(photo)
            if not metadata:skips.append({'photo':photo,'reason':'missing exposure metadata'});continue
            x=morph.load(photo,a.size);y=morph.load(Path(a.edits)/Path(photo).name,a.size);baseline=gamut_renderer.render(x,look);original=float(np.sqrt(np.mean((baseline-y)**2)));ratio=math.log2(metadata/ref);row={'scene':scene['id'],'photo':photo,'metadata_stop_difference':ratio,'baseline_rmse':original,'trials':{}}
            for factor in (.25,.5,1.):
                correction=float(np.clip(-ratio*factor,-.5,.5));initial=gamut_renderer.vec({'exposure':correction});fixed=gamut_renderer.render(gamut_renderer.render(x,initial),look);row['trials'][str(factor)]={'correction':correction,'rmse':float(np.sqrt(np.mean((fixed-y)**2)))}
            rows.append(row)
        print(scene['id'],flush=True)
    summary={}
    for factor in (.25,.5,1.):
        before=np.array([r['baseline_rmse'] for r in rows]);after=np.array([r['trials'][str(factor)]['rmse'] for r in rows]);delta=after-before
        summary[str(factor)]={'pairs':len(rows),'improved':int(np.sum(delta< -1e-6)),'worsened':int(np.sum(delta>1e-6)),'unchanged_within_1e-6':int(np.sum(np.abs(delta)<=1e-6)),'baseline_mean':float(np.mean(before)),'corrected_mean':float(np.mean(after)),'baseline_median':float(np.median(before)),'corrected_median':float(np.median(after))}
    morph.dump({'scope':'Exploratory camera shutter*ISO/f-number^2 proxy relative to the scene representative, followed by a clamped per-photo correction before the fitted look. Metadata does not measure scene light or intent. Same dataset and unvalidated groups; no unseen album.','summary':summary,'rows':rows,'skipped':skips},a.out)
if __name__=='__main__':main()
