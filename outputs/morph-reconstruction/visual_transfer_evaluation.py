#!/usr/bin/env python3
"""Evaluate visual-nearest reference selection against fixed chronological references."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import morph,poly_color


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('sensitivity');p.add_argument('grouping');p.add_argument('distances');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args()
 study=json.loads(Path(a.sensitivity).read_text());groups=json.loads(Path(a.grouping).read_text());features=json.loads(Path(a.distances).read_text());paths=features['paths'];index={str(Path(x).resolve()):i for i,x in enumerate(paths)};matrix=np.asarray(features['distances'],dtype=float)
 if len(index)!=len(paths) or matrix.shape!=(len(paths),len(paths)) or not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.T) or not np.allclose(np.diag(matrix),0):raise ValueError('Invalid distance matrix')
 group_index={str(Path(p).resolve()):g['id'] for g in groups['groups'] for p in g['photos']}
 if set(index)!=set(group_index):raise ValueError('Grouping and feature paths differ')
 manifest=groups.get('source_manifest_sha256',{})
 if not manifest or any(morph.sha256_file(path)!=digest for path,digest in manifest.items()):raise ValueError('Missing or changed feature-extraction source files')
 edit_paths={str(Path(r['photo']).resolve()):Path(a.edits)/Path(r['photo']).name for s in study['scenes'] for r in s['heldout']};edit_hashes={str(p):morph.sha256_file(p) for p in edit_paths.values()};images={};weights={}
 def pair(path):
  if path not in images:
   with Image.open(path) as im:source_size=ImageOps.exif_transpose(im).size
   with Image.open(edit_paths[path]) as im:edit_size=ImageOps.exif_transpose(im).size
   if source_size!=edit_size:raise ValueError('Input and edited image dimensions differ: '+path)
   x=morph.load(path,study['size']);y=morph.load(edit_paths[path],study['size'])
   if x.shape!=y.shape:raise ValueError('Input and edited image dimensions differ: '+path)
   images[path]=(x,y)
  return images[path]
 def score(rep,target):
  if rep not in weights:
   x,y=pair(rep);weights[rep]=poly_color.fit(x,y,regularization=study['regularization'])[0]
  x,y=pair(target);return float(np.sqrt(np.mean((poly_color.render(x,weights[rep])-y)**2)))
 rows=[];skipped=[]
 for scene in study['scenes']:
  fixed=str(Path(scene['heldout'][0]['photo']).resolve())
  for held in scene['heldout']:
   target=str(Path(held['photo']).resolve())
   if target==fixed:continue
   candidates=[(str(Path(r['representative']).resolve()),float(r['rmse'])) for r in held['representatives']]
   if target not in index or any(p not in index for p,_ in candidates):raise ValueError('Photo missing from features')
   in_group=[v for v in candidates if group_index[v[0]]==group_index[target]]
   fixed_error=score(fixed,target)
   nearest=min(candidates,key=lambda item:(matrix[index[target],index[item[0]]],item[0]))
   x,y=pair(target);row={'scene':scene['scene'],'photo':target,'proposed_group':group_index[target],'fixed_representative':fixed,'fixed_rmse':fixed_error,'nearest_representative':nearest[0],'nearest_distance':float(matrix[index[target],index[nearest[0]]]),'nearest_rmse':score(nearest[0],target),'source_rmse':float(np.sqrt(np.mean((x-y)**2)))}
   if in_group:
    same=min(in_group,key=lambda item:(matrix[index[target],index[item[0]]],item[0]));row.update({'same_group_representative':same[0],'same_group_rmse':score(same[0],target)})
   else:skipped.append({'scene':scene['scene'],'photo':target,'reason':'no other eligible edited pair in visual subgroup'})
   rows.append(row)
 if not rows:raise ValueError('No held-out comparisons')
 def summary(subset,field):
  fixed=np.array([r['fixed_rmse'] for r in subset]);chosen=np.array([r[field] for r in subset]);return {'photos':len(subset),'fixed_mean_rmse':float(fixed.mean()),'chosen_mean_rmse':float(chosen.mean()),'chosen_lower':int(np.sum(chosen<fixed-1e-8)),'chosen_higher':int(np.sum(chosen>fixed+1e-8)),'equal':int(np.sum(np.abs(chosen-fixed)<=1e-8))}
 subset=[r for r in rows if 'same_group_rmse' in r];out={'scope':'Exploratory one-album comparison on preselected same-dimension human-edited pairs. Fixed representative is the first photo from each sensitivity scene; held-out photos never provide their own training edit. Visual choices require an independently edited eligible reference and select it based only on distances between originals. Same-group selection excludes held-outs with no other eligible edited pair. All methods use polynomial fits at the same size; groups and pair alignment are not verified, edits may have different intentions. Threshold and features were inspected on this album; this is not independent validation. RGB RMSE is not artistic quality.','size':study['size'],'nearest_within_chronological_scene':summary(rows,'nearest_rmse'),'nearest_within_visual_subgroup':summary(subset,'same_group_rmse') if subset else None,'excluded_from_same_group':skipped,'photos':rows}
 if any(morph.sha256_file(path)!=digest for path,digest in edit_hashes.items()):raise ValueError('Edited inputs changed during evaluation')
 out['edit_sha256']=edit_hashes;morph.protect_inputs([a.sensitivity,a.grouping,a.distances]+list(index)+list(edit_hashes),[a.out]);morph.protect_existing([a.out],False);morph.dump(out,a.out);print(json.dumps({'nearest':out['nearest_within_chronological_scene'],'same_group':out['nearest_within_visual_subgroup'],'excluded':len(skipped)},indent=2))
if __name__=='__main__':main()
