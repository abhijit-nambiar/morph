#!/usr/bin/env python3
"""Compare preselected scene split against a synthetic target on held-out photos."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,face_core


def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('unsplit_fit');p.add_argument('split_fits',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    old=json.loads(Path(a.unsplit_fit).read_text());splits=[json.loads(Path(f).read_text()) for f in a.split_fits]
    if old.get('target_engine')!='synthetic-face' or any(s.get('target_engine')!='synthetic-face' for s in splits):raise ValueError('Requires synthetic face fits')
    if any(s['target_settings']!=old['target_settings'] or s['target_face_light']!=old['target_face_light'] or s['fit_size']!=old['fit_size'] for s in splits):raise ValueError('Target settings or fitting size differ')
    original={r['path']:r for r in old['manifest']};members={}
    for split in splits:
        for record in split['manifest']:
            path=record['path']
            if path in members or path not in original:raise ValueError('Invalid split membership')
            before=original[path]
            if record['sha256']!=before['sha256'] or record['faces']!=before['faces']:raise ValueError('Source or detections differ')
            members[path]=(split,record)
    if set(members)!=set(original):raise ValueError('Split does not preserve exact membership')
    for path,record in original.items():
        if morph.sha256_file(path)!=record['sha256']:raise ValueError('Photo changed since fitting')
    reps={old['representative']}|{s['representative'] for s in splits};rows=[]
    for path,(fit,record) in members.items():
        if path in reps:continue
        if morph.sha256_file(path)!=record['sha256']:raise ValueError('Photo changed since fitting')
        x=morph.load(path,old['fit_size']);boxes=record['faces'];target=face_core.target(x,boxes,old['target_settings'],old['target_face_light'])
        before=face_core.render(x,boxes,np.asarray(old['weights']));after=face_core.render(x,boxes,np.asarray(fit['weights']))
        rows.append({'photo':path,'split_scene':fit['scene'],'source_rmse':rmse(x,target),'unsplit_rmse':rmse(before,target),'split_rmse':rmse(after,target)})
    if not rows:raise ValueError('No held-out photos')
    before=np.array([r['unsplit_rmse'] for r in rows]);after=np.array([r['split_rmse'] for r in rows]);tol=1e-6
    morph.protect_inputs([a.unsplit_fit,*a.split_fits,*members],[a.out]);morph.protect_existing([a.out],False)
    morph.dump({'scope':'The split was selected by visual inspection of this same album. Synthetic target uses the same global controls and face detections for every photo and is not a human edit; both methods share the target generator. Every representative used by either method is excluded from evaluation. Pixel RMSE does not measure aesthetics or validate grouping in other albums.','representatives_excluded':sorted(reps),'summary':{'heldout':len(rows),'split_lower':int(np.sum(after<before-tol)),'split_higher':int(np.sum(after>before+tol)),'effectively_equal':int(np.sum(abs(after-before)<=tol)),'unsplit_mean_rmse':float(before.mean()),'split_mean_rmse':float(after.mean()),'unsplit_median_rmse':float(np.median(before)),'split_median_rmse':float(np.median(after))},'photos':rows},a.out)
    print(len(rows),'held-out photos')
if __name__=='__main__':main()
