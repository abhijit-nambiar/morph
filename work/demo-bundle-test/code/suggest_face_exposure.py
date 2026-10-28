#!/usr/bin/env python3
"""Draft review-only metadata exposure suggestions for face scene fits."""
import argparse,json,math
from pathlib import Path
import morph,face_scene_pipeline,suggest_exposure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits');p.add_argument('--out',required=True);p.add_argument('--scale',type=float,default=.25);p.add_argument('--max-correction',type=float,default=.25);p.add_argument('--min-difference',type=float,default=.3);p.add_argument('--max-difference',type=float,default=2.);p.add_argument('--max-minutes',type=float,default=15.);a=p.parse_args()
    vals=(a.scale,a.max_correction,a.min_difference,a.max_difference,a.max_minutes)
    if not all(math.isfinite(v) for v in vals) or not (0<a.scale<=1 and 0<a.max_correction<=.5 and 0<=a.min_difference<a.max_difference and a.max_minutes>0):raise ValueError('Invalid thresholds')
    fits=Path(a.fits);index=json.loads((fits/'index.json').read_text());scenes=index.get('scenes')
    if index.get('version')!=1 or not isinstance(scenes,list) or not scenes or len(set(scenes))!=len(scenes):raise ValueError('Invalid album fit index')
    plans=[];inputs=[fits/'index.json']
    for scene in scenes:
        if not isinstance(scene,str) or not __import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',scene):raise ValueError('Invalid scene ID')
        fitfile=fits/scene/'fit.json';fit,_,sources=face_scene_pipeline.validate_fit(fitfile)
        if fit.get('scene')!=scene:raise ValueError('Scene mismatch')
        inputs.append(fitfile);inputs.extend(path for path,_ in sources);plans.append((scene,fit,sources))
    out=Path(a.out);audit=Path(str(out)+'.audit.json');morph.protect_inputs(inputs,[out,audit]);morph.protect_existing([out,audit],False)
    records=[];proposed={}
    for scene,fit,sources in plans:
        reference=suggest_exposure.metadata(fit['representative']);changes={}
        for path,_ in sources:
            item=suggest_exposure.propose(reference,suggest_exposure.metadata(path),a.scale,a.max_correction,a.min_difference,a.max_difference,a.max_minutes*60)
            records.append({'scene':scene,'photo':path,**item})
            if item['eligible']:changes[Path(path).name]={'exposure':item['proposed_exposure']}
        if changes:proposed[scene]=changes
    morph.dump({'photos':proposed},out);morph.dump({'warning':'Review candidates; camera settings do not measure scene light, JPEG processing, or photographer intent. This heuristic has mixed results and is not automatically applied.','photos':records},audit)
    print(sum(x['eligible'] for x in records),'review candidates among',len(records),'photos')
if __name__=='__main__':main()
