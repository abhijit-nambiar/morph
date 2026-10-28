#!/usr/bin/env python3
"""Flag scene photos with unusual detected-face brightness for manual review."""
import argparse,json,math
from pathlib import Path
import numpy as np
import morph,face_scene_pipeline


def face_brightness(x,boxes):
    """Median of per-box Gaussian-weighted sRGB luminances; not skin segmentation."""
    h,w=x.shape[:2];yy=(np.arange(h,dtype=np.float32)+.5)/h;xx=(np.arange(w,dtype=np.float32)+.5)/w
    lum=np.sum(x*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1);values=[]
    for b in boxes:
        if b['width']*w<3 or b['height']*h<3:continue
        cx=b['x']+b['width']*.5;cy=b['y']+b['height']*.5;sx=b['width']*.35;sy=b['height']*.35
        mask=np.exp(-.5*(((xx[None,:]-cx)/sx)**2+((yy[:,None]-cy)/sy)**2));total=float(mask.sum())
        if total>1e-6:values.append(float(np.sum(mask*lum)/total))
    return (float(np.median(values)) if values else None),len(values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits');p.add_argument('album');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=512);p.add_argument('--threshold',type=float,default=.5,help='Absolute log2 luminance ratio for review flag; not camera exposure stops');a=p.parse_args()
    if not 64<=a.size<=1024 or not math.isfinite(a.threshold) or not .05<=a.threshold<=3:raise ValueError('Invalid sample size or threshold')
    fits=Path(a.fits);album=Path(a.album);index=json.loads((fits/'index.json').read_text());report=json.loads((album/'report.json').read_text())
    if report['scenes']!=index['scenes']:raise ValueError('Scene lists do not match')
    inputs=[fits/'index.json',album/'report.json'];plans=[]
    for scene in index['scenes']:
        fitfile=fits/scene/'fit.json';fit,_,sources=face_scene_pipeline.validate_fit(fitfile);scene_report=album/scene/'report.json';review=json.loads(scene_report.read_text());inputs.extend([fitfile,scene_report]);by_path=dict(sources)
        if fit['scene']!=scene or review['scene']!=scene or Path(review['source_fit']).resolve()!=fitfile.resolve():raise ValueError('Scene report references a different fit')
        seen=set()
        for item in review['photos']:
            source=str(Path(item['source']).resolve());output=Path(item['output'])
            if source not in by_path or source in seen:raise ValueError('Unknown or duplicate source photo')
            seen.add(source);inputs.extend([source,output]);plans.append((scene,source,str(output),by_path[source]))
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False);rows=[]
    for scene,source,output,boxes in plans:
        before,count=face_brightness(morph.load(source,a.size),boxes);after,out_count=face_brightness(morph.load(output,a.size),boxes)
        if count!=out_count:raise ValueError('Input/output face sampling differs')
        rows.append({'scene':scene,'source':source,'export':output,'detected_faces':len(boxes),'measured_faces':count,'source_face_luminance':before,'output_face_luminance':after})
    summaries=[]
    for scene in index['scenes']:
        subset=[r for r in rows if r['scene']==scene and r['output_face_luminance'] is not None]
        median=float(np.median([r['output_face_luminance'] for r in subset])) if len(subset)>=3 else None
        for row in subset:
            if median is None:row['log2_luminance_ratio_to_scene_median']=None;row['review_flag']=None
            else:
                ratio=math.log2(max(row['output_face_luminance'],1e-5)/max(median,1e-5));row['log2_luminance_ratio_to_scene_median']=ratio;row['review_flag']=abs(ratio)>a.threshold
        summaries.append({'scene':scene,'photos':sum(r['scene']==scene for r in rows),'measured_photos':len(subset),'status':'evaluated' if median is not None else 'insufficient measured photos','median_output_face_luminance':median,'review_flags':sum(r.get('review_flag') is True for r in subset)})
    morph.dump({'scope':'Reduced-resolution Gaussian-weighted RGB luminance around detected face boxes, summarized by median per photo and compared with its scene median. This is a review heuristic, not an exposure measurement, skin mask, identity-matched comparison, or quality verdict. Detected boxes may include background or miss faces. Different people, skin tones, lighting and intent affect the measure; no automatic corrections are made.','size':a.size,'threshold_log2_luminance_ratio':a.threshold,'mode':report['mode'],'summary':{'photos':len(rows),'measured':sum(r['measured_faces']>0 for r in rows),'unmeasured':sum(r['measured_faces']==0 for r in rows),'flags':sum(r.get('review_flag') is True for r in rows)},'scenes':summaries,'photos':rows},a.out)
    print(len(rows),'photos;',sum(r.get('review_flag') is True for r in rows),'review flags;',sum(r['measured_faces']==0 for r in rows),'unmeasured')
if __name__=='__main__':main()
