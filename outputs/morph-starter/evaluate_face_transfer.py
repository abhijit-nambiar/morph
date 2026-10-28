#!/usr/bin/env python3
"""Evaluate synthetic face-scene target reconstruction on rendered photos."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,face_core,face_scene_pipeline,gamut_renderer


def rmse(x,y):return float(np.sqrt(np.mean((x-y)**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits');p.add_argument('album');p.add_argument('--out',required=True);a=p.parse_args()
    fits=Path(a.fits);album=Path(a.album);index=json.loads((fits/'index.json').read_text());report=json.loads((album/'report.json').read_text())
    if report['scenes']!=index['scenes']:raise ValueError('Scene lists do not match')
    inputs=[fits/'index.json',album/'report.json'];plans=[]
    for scene in index['scenes']:
        fitfile=fits/scene/'fit.json';fit,w,sources=face_scene_pipeline.validate_fit(fitfile)
        if fit['scene']!=scene or fit.get('target_engine')!='synthetic-face':raise ValueError('Only synthetic face target fits are supported: '+scene)
        source_map=dict(sources);scene_report=album/scene/'report.json';data=json.loads(scene_report.read_text());inputs.extend([fitfile,scene_report]);cor=data.get('corrections',{})
        if data.get('scene')!=scene or Path(data['source_fit']).resolve()!=fitfile.resolve():raise ValueError('Rendered report references a different fit: '+scene)
        seen=set()
        for item in data['photos']:
            path=str(Path(item['source']).resolve())
            if path not in source_map or path in seen:raise ValueError('Photo missing or duplicated in fit: '+path)
            seen.add(path);inputs.append(path);plans.append((scene,path,source_map[path],fit,w,cor))
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False);rows=[]
    for scene,path,boxes,fit,w,cor in plans:
        x=morph.load(path,fit['fit_size']);target=face_core.target(x,boxes,fit['target_settings'],fit['target_face_light'])
        photo=gamut_renderer.vec(cor.get('photos',{}).get(Path(path).name,{}));scene_c=gamut_renderer.vec(cor.get('scene',{}));album_c=gamut_renderer.vec(cor.get('album',{}))
        output=gamut_renderer.render(x,photo) if np.any(photo) else x;output=face_core.render(output,boxes,w)
        if np.any(scene_c):output=gamut_renderer.render(output,scene_c)
        if np.any(album_c):output=gamut_renderer.render(output,album_c)
        baseline=rmse(x,target);fitted=rmse(output,target)
        rows.append({'scene':scene,'photo':path,'role':'representative' if path==fit['representative'] else 'other photo','faces':len(boxes),'source_rmse':baseline,'output_rmse':fitted,'worse_than_source':fitted>baseline+1e-6})
    if not rows:raise ValueError('No rendered photos')
    held=[r for r in rows if r['role']=='other photo'];base=np.array([r['source_rmse'] for r in held]);after=np.array([r['output_rmse'] for r in held])
    summary={'rendered':len(rows),'representatives':len(rows)-len(held),'other_photos':len(held),'other_photos_worse_than_source':sum(r['worse_than_source'] for r in held),'source_mean_rmse':float(base.mean()) if len(base) else None,'output_mean_rmse':float(after.mean()) if len(after) else None,'output_max_rmse':float(after.max()) if len(after) else None}
    morph.dump({'scope':'Synthetic target independently recomputed per photo using the same target generator and detected face boxes used during fitting; target and model masks differ in shape. Scores are reduced-resolution, pre-JPEG RGB RMSE; this is not a visual-quality score or evaluation against human edits. The scene grouping and face detections are unverified.','mode':report['mode'],'summary':summary,'photos':rows},a.out)
    print(len(held),'other photos;',summary['other_photos_worse_than_source'],'worse than source')
if __name__=='__main__':main()
