#!/usr/bin/env python3
"""Evaluate a conservative face-weighted exposure heuristic on real held-out pairs."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,gamut_renderer,face_light_experiment as face,suggest_exposure


def face_luminance(x,boxes):
    mask=face.mask(x.shape,boxes);lum=np.sum(x*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1)
    if float(mask.sum())<1:raise ValueError('No usable face mask')
    return float(np.sum(lum*mask)/mask.sum())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline');p.add_argument('detections');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--gain',type=float,default=.5);p.add_argument('--limit',type=float,default=.25);p.add_argument('--deadband',type=float,default=.08);a=p.parse_args()
    if not np.isfinite(a.gain) or not 0<=a.gain<=1 or not np.isfinite(a.limit) or not 0<=a.limit<=.5 or not np.isfinite(a.deadband) or not 0<=a.deadband<=1:raise ValueError('Invalid heuristic parameters')
    report=json.loads(Path(a.baseline).read_text());lookup={str(Path(x['path']).resolve()):x for x in json.loads(Path(a.detections).read_text())};rows=[]
    for scene in report['scenes']:
        rep=scene['representative'];reference=morph.load(rep,report['size']);reference_lum=face_luminance(reference,lookup[rep]['faces']);weights=np.asarray(scene['face_weights']);ref_metadata=suggest_exposure.metadata(rep)
        for entry in scene['heldout']:
            path=entry['photo'];x=morph.load(path,report['size']);y=morph.load(Path(a.edits)/Path(path).name,report['size']);boxes=lookup[path]['faces'];lum=face_luminance(x,boxes)
            delta=np.log2(max(reference_lum,1e-5)/max(lum,1e-5));adjustment=0. if abs(delta)<a.deadband else float(np.clip(a.gain*delta,-a.limit,a.limit))
            corrected=gamut_renderer.render(x,gamut_renderer.vec({'exposure':adjustment}));result=face.render(corrected,boxes,weights)
            metadata=suggest_exposure.propose(ref_metadata,suggest_exposure.metadata(path));agreement=bool(metadata.get('eligible') and adjustment*metadata['proposed_exposure']>0)
            rows.append({'scene':scene['scene'],'photo':path,'metadata_suggestion':metadata,'metadata_direction_agrees':agreement,'representative_face_luminance':reference_lum,'photo_face_luminance':lum,'log2_ratio':float(delta),'correction':adjustment,'baseline_rmse':entry['face_rmse'],'corrected_rmse':face.rmse(result,y)})
        print(scene['scene'],len(scene['heldout']),flush=True)
    if not rows:raise ValueError('No held-out photos')
    b=np.array([r['baseline_rmse'] for r in rows]);c=np.array([r['corrected_rmse'] for r in rows]);diff=c-b;tol=1e-6
    agreed=np.array([r['corrected_rmse']-r['baseline_rmse'] for r in rows if r['metadata_direction_agrees']]);agreement_summary={'cases':len(agreed),'improved':int(np.sum(agreed < -tol)),'worsened':int(np.sum(agreed > tol)),'unchanged':int(np.sum(abs(agreed)<=tol)),'mean_rmse_change':float(agreed.mean()) if len(agreed) else None}
    morph.dump({'metadata_agreement_scope':'Exploratory post-hoc filter using default suggest_exposure metadata eligibility and matching correction direction; selection was chosen after inspecting the data. Not independent validation.', 'metadata_agreement_summary':agreement_summary,'scope':'Face-weighted luminance heuristic uses the same unverified face boxes as the face mapping; approximate face regions include surroundings, and complexion, composition, lighting, or artistic intent can differ. Fixed heuristic settings, one album, held-out photos, same reduced-resolution real targets; alignment and quality unverified.','gain':a.gain,'limit':a.limit,'deadband_log2':a.deadband,'summary':{'heldout':len(rows),'corrected':sum(abs(r['correction'])>0 for r in rows),'improved':int(np.sum(diff < -tol)),'worsened':int(np.sum(diff > tol)),'unchanged':int(np.sum(abs(diff)<=tol)),'baseline_mean_rmse':float(b.mean()),'corrected_mean_rmse':float(c.mean()),'baseline_median_rmse':float(np.median(b)),'corrected_median_rmse':float(np.median(c))},'rows':rows},a.out)
if __name__=='__main__':main()
