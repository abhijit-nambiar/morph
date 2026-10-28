#!/usr/bin/env python3
"""Compare face-relative and frame-relative mappings on held-out human edits."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,face_light_experiment as face


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('split_report');p.add_argument('grid_report');p.add_argument('detections');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--regularization',type=float,default=.001);a=p.parse_args()
    split=json.loads(Path(a.split_report).read_text());grid=json.loads(Path(a.grid_report).read_text());detections=json.loads(Path(a.detections).read_text());lookup={str(Path(r['path']).resolve()):r for r in detections};gridrows={(r['scene'],str(Path(r['photo']).resolve())):r for r in grid['cases']};rows=[];scenes=[];skipped=[]
    if a.size!=split['size'] or a.regularization!=split['regularization']:raise ValueError('Use the same size and regularization as the baseline report')
    for s in split['scenes']:
        rep=str(Path(s['representative']).resolve());d=lookup.get(rep)
        if d is None or d.get('error') or not d['faces']:
            skipped.append({'scene':s['id'],'reason':'representative detection unavailable'});continue
        x=morph.load(rep,a.size);y=morph.load(Path(a.edits)/Path(rep).name,a.size);w=face.fit(x,y,d['faces'],penalty=a.regularization);held=[]
        for row in s['heldout']:
            photo=str(Path(row['photo']).resolve());d=lookup.get(photo);baseline=gridrows[(s['id'],photo)]
            if d is None or d.get('error') or not d['faces']:
                skipped.append({'scene':s['id'],'photo':photo,'reason':'detection unavailable'});continue
            x=morph.load(photo,a.size);y=morph.load(Path(a.edits)/Path(photo).name,a.size);raw=x+face.features(x,d['faces'])@w
            result={'scene':s['id'],'photo':photo,'faces':len(d['faces']),'source_rmse':face.rmse(x,y),'global_rmse':baseline['global_rmse'],'grid_rmse':baseline['grid_rmse'],'face_rmse':face.rmse(np.clip(raw,0,1),y),'face_out_of_gamut_fraction':float(np.mean(np.any((raw<0)|(raw>1),axis=-1)))};rows.append(result);held.append(result)
        scenes.append({'scene':s['id'],'representative':rep,'face_weights':w.tolist(),'heldout':held});print(s['id'],len(held),'held-out',flush=True)
    if not rows:raise ValueError('No eligible photos')
    g=np.array([r['global_rmse'] for r in rows]);grid=np.array([r['grid_rmse'] for r in rows]);f=np.array([r['face_rmse'] for r in rows]);source=np.array([r['source_rmse'] for r in rows])
    morph.dump({'scope':'One album, same representative and held-out pairs as the previous grid comparison, same deterministic fitting sample and regularization. Face boxes detected locally and not independently validated. Equal image dimensions do not guarantee alignment and edits may have different intentions. RGB RMSE is not photographic quality.','size':a.size,'regularization':a.regularization,'summary':{'scenes':len(scenes),'heldout':len(rows),'face_lower_than_grid':int(np.sum(f<grid)),'face_lower_than_global':int(np.sum(f<g)),'face_lower_than_source':int(np.sum(f<source)),'source_mean_rmse':float(source.mean()),'global_mean_rmse':float(g.mean()),'grid_mean_rmse':float(grid.mean()),'face_mean_rmse':float(f.mean()),'face_median_rmse':float(np.median(f)),'face_mean_out_of_gamut_fraction':float(np.mean([r['face_out_of_gamut_fraction'] for r in rows]))},'skipped':skipped,'scenes':scenes},a.out)
if __name__=='__main__':main()
