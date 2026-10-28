#!/usr/bin/env python3
"""Compare brightness review flags before and after a proposed scene split."""
import argparse,json,math
from pathlib import Path
import numpy as np
import morph,face_core,audit_scene_consistency


def readfit(path):
    f=json.loads(Path(path).read_text());return f,{r['path']:r for r in f['manifest']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('original_fit');p.add_argument('split_fits',nargs='+');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=320);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
    if not 64<=a.size<=1024 or not math.isfinite(a.threshold) or not .05<=a.threshold<=3:raise ValueError('Invalid size or threshold')
    original,old=readfit(a.original_fit);splits=[readfit(x) for x in a.split_fits];new={}
    for fit,manifest in splits:
        for path,r in manifest.items():
            if path in new or path not in old:raise ValueError('Invalid membership')
            if r['faces']!=old[path]['faces'] or r['sha256']!=old[path]['sha256']:raise ValueError('Face boxes or source checksum differ')
            new[path]=(fit,r)
    if set(new)!=set(old):raise ValueError('Split does not preserve all photos')
    if any(f['target_settings']!=original['target_settings'] or f['target_face_light']!=original['target_face_light'] for f,_ in splits):raise ValueError('Fit target settings differ')
    inputs=[a.original_fit,*a.split_fits,*old];morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False);rows=[]
    for path,(fit,r) in new.items():
        if morph.sha256_file(path)!=r['sha256']:raise ValueError('Source changed: '+path)
        x=morph.load(path,a.size);before=face_core.render(x,r['faces'],np.array(original['weights']));after=face_core.render(x,r['faces'],np.array(fit['weights']));b,n=audit_scene_consistency.face_brightness(before,r['faces']);c,m=audit_scene_consistency.face_brightness(after,r['faces'])
        rows.append({'photo':path,'split_scene':fit['scene'],'measured_faces':n,'old_luminance':b,'new_luminance':c})
    old_values=[r['old_luminance'] for r in rows if r['old_luminance'] is not None];old_median=float(np.median(old_values)) if len(old_values)>=3 else None
    split_medians={}
    for fit,_ in splits:
        vals=[r['new_luminance'] for r in rows if r['split_scene']==fit['scene'] and r['new_luminance'] is not None];split_medians[fit['scene']]=float(np.median(vals)) if len(vals)>=3 else None
    for r in rows:
        for prefix,value,median in [('old',r['old_luminance'],old_median),('new',r['new_luminance'],split_medians[r['split_scene']])]:
            ratio=math.log2(max(value,1e-5)/max(median,1e-5)) if value is not None and median is not None else None
            r[prefix+'_log2_luminance_ratio']=ratio;r[prefix+'_review_flag']=abs(ratio)>a.threshold if ratio is not None else None
    morph.dump({'scope':'Exploratory same-album, reduced-resolution pre-JPEG brightness proxy using the same detections for both fits. Group boundaries, representatives and mapping change together. Face boxes include surrounding pixels and can miss people; brightness varies with identity, pose, background and artistic intent. Flags do not establish quality or a need for correction.','threshold_log2_luminance_ratio':a.threshold,'size':a.size,'old_scene_median':old_median,'split_scene_medians':split_medians,'old_flags':sum(r['old_review_flag'] is True for r in rows),'new_flags':sum(r['new_review_flag'] is True for r in rows),'photos':rows},a.out)
    print(len(rows),'photos;',sum(r['old_review_flag'] is True for r in rows),'old flags;',sum(r['new_review_flag'] is True for r in rows),'split flags')
if __name__=='__main__':main()
