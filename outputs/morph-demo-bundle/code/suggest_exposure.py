#!/usr/bin/env python3
"""Draft conservative, review-only exposure suggestions from camera settings."""
import argparse,json,math
from datetime import datetime
from pathlib import Path
from PIL import Image
import morph

def metadata(path):
    with Image.open(path) as im:
        e=im.getexif();d=e.get_ifd(34665);camera=e.get(272);timestamp=d.get(36867) or e.get(306);shutter=d.get(33434);aperture=d.get(33437);iso=d.get(34855)
    try:
        if isinstance(iso,(tuple,list)):iso=iso[0]
        timestamp=datetime.strptime(timestamp,'%Y:%m:%d %H:%M:%S');shutter=float(shutter);aperture=float(aperture);iso=float(iso)
        if not camera or min(shutter,aperture,iso)<=0 or not all(math.isfinite(v) for v in (shutter,aperture,iso)):return None
        return {'camera':str(camera),'timestamp':timestamp,'proxy':shutter*iso/(aperture*aperture)}
    except (ValueError,TypeError,ZeroDivisionError):return None

def propose(ref,current,scale=.25,limit=.25,min_delta=.3,max_delta=2,max_seconds=900):
    if ref is None or current is None:return {'eligible':False,'reason':'missing camera, time, shutter, aperture, or ISO'}
    if ref['camera']!=current['camera']:return {'eligible':False,'reason':'camera model differs'}
    elapsed=abs((ref['timestamp']-current['timestamp']).total_seconds())
    if elapsed>max_seconds:return {'eligible':False,'reason':'capture time too far from representative','elapsed_seconds':elapsed}
    delta=math.log2(current['proxy']/ref['proxy'])
    if abs(delta)<min_delta:return {'eligible':False,'reason':'camera exposure difference below threshold','metadata_stops':delta}
    if abs(delta)>max_delta:return {'eligible':False,'reason':'camera exposure difference above threshold','metadata_stops':delta}
    correction=max(-limit,min(limit,-delta*scale))
    return {'eligible':True,'reason':'review candidate, not an automatic correction','metadata_stops':delta,'elapsed_seconds':elapsed,'proposed_exposure':correction}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('fits');p.add_argument('--out',required=True);p.add_argument('--scale',type=float,default=.25);p.add_argument('--max-correction',type=float,default=.25);p.add_argument('--min-difference',type=float,default=.3);p.add_argument('--max-difference',type=float,default=2);p.add_argument('--max-minutes',type=float,default=15);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    if not all(math.isfinite(v) for v in (a.scale,a.max_correction,a.min_difference,a.max_difference,a.max_minutes)) or not (0<a.scale<=1 and 0<a.max_correction<=.5 and 0<=a.min_difference<a.max_difference and a.max_minutes>0):p.error('Invalid thresholds')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));inputs=[a.groups]+[p for g in groups for p in g['photos']]+[Path(a.fits)/g['id']/'fit.json' for g in groups];out=Path(a.out);audit=Path(str(out)+'.audit.json');morph.protect_inputs(inputs,[out,audit]);morph.protect_existing([out,audit],a.overwrite)
    proposal={'photos':{}};records=[]
    for g in groups:
        fit=json.loads((Path(a.fits)/g['id']/'fit.json').read_text())
        if fit.get('renderer','global') not in ('gamut','polynomial','spatial','grid'):raise ValueError('Exposure suggestions require gamut, polynomial, spatial, or grid fits: '+g['id'])
        if Path(fit['representative']).resolve()!=Path(g['representative']).resolve():raise ValueError('Representative differs from saved fit: '+g['id'])
        ref=metadata(g['representative']);changes={}
        for photo in g['photos']:
            result=propose(ref,metadata(photo),a.scale,a.max_correction,a.min_difference,a.max_difference,a.max_minutes*60);records.append({'scene':g['id'],'photo':photo,**result})
            if result['eligible']:changes[Path(photo).name]={'exposure':result['proposed_exposure']}
        if changes:proposal['photos'][g['id']]=changes
    morph.dump(proposal,out);morph.dump({'warning':'Review suggestions before applying. Camera exposure settings alone do not measure scene lighting, intent, or JPEG processing. This method was tested on one album with mixed results.','method':{'scale':a.scale,'max_correction':a.max_correction,'min_difference_stops':a.min_difference,'max_difference_stops':a.max_difference,'max_minutes':a.max_minutes},'suggestions':sum(x['eligible'] for x in records),'photos':records},audit);print('Review candidates:',sum(x['eligible'] for x in records),'of',len(records),'photos; no images were changed.')
if __name__=='__main__':main()
