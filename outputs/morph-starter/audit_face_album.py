#!/usr/bin/env python3
"""Audit reduced-resolution pre-JPEG clipping for an existing face-album render."""
import argparse,json
from pathlib import Path
import morph,face_scene_pipeline,face_quality


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits');p.add_argument('album');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=512);a=p.parse_args()
    if not 64<=a.size<=1024:raise ValueError('size must be 64..1024')
    fits=Path(a.fits);album=Path(a.album);index=json.loads((fits/'index.json').read_text());report=json.loads((album/'report.json').read_text());scenes=index['scenes']
    if report['scenes']!=scenes:raise ValueError('Album scenes differ from fits')
    inputs=[fits/'index.json',album/'report.json'];plans=[]
    for scene in scenes:
        fitfile=fits/scene/'fit.json';fit,w,sources=face_scene_pipeline.validate_fit(fitfile);by_path=dict(sources);scene_report=album/scene/'report.json';review=json.loads(scene_report.read_text());inputs.extend([fitfile,scene_report])
        if fit['scene']!=scene or review['scene']!=scene:raise ValueError('Scene mismatch')
        corrections=review.get('corrections',{});album_fix=corrections.get('album',{});scene_fix=corrections.get('scene',{});photos_fix=corrections.get('photos',{})
        for item in review['photos']:
            path=str(Path(item['source']).resolve())
            if path not in by_path:raise ValueError('Reported source absent from fit')
            inputs.append(path);plans.append((scene,path,by_path[path],w,album_fix,scene_fix,photos_fix.get(Path(path).name,{})))
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False);rows=[]
    for scene,path,boxes,w,album_fix,scene_fix,photo_fix in plans:
        x=morph.load(path,a.size);d=face_quality.diagnose(x,w,boxes,album_fix,scene_fix,photo_fix);rows.append({'scene':scene,'source':path,'faces':len(boxes),'clipping':d})
    morph.dump({'scope':'Approximate reduced-resolution pre-JPEG check, recomputed from originals and stored settings. Downsampling may hide localized clipping or artifacts; near-limit increases are heuristics, not judgments of quality. Out-of-gamut fraction is measured before the fitted mapping clamp, after any photo correction and before later corrections.','sample_max_side':a.size,'photos':len(rows),'near_limit_flags':sum(r['clipping']['near_limit_flag'] for r in rows),'photos_without_faces':sum(not r['faces'] for r in rows),'rows':rows},a.out)
    print(len(rows),'photos;',sum(r['clipping']['near_limit_flag'] for r in rows),'near-limit flags')
if __name__=='__main__':main()
