#!/usr/bin/env python3
"""Compare one mapping per proposed visual group with one per chronological scene."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import morph,poly_color


def main():
 p=argparse.ArgumentParser();p.add_argument('sensitivity');p.add_argument('grouping');p.add_argument('distances');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args()
 study=json.loads(Path(a.sensitivity).read_text());groupdata=json.loads(Path(a.grouping).read_text());data=json.loads(Path(a.distances).read_text());paths=[str(Path(p).resolve()) for p in data['paths']];d=np.asarray(data['distances'],dtype=float);lookup={p:i for i,p in enumerate(paths)}
 if len(lookup)!=len(paths) or d.shape!=(len(paths),len(paths)) or not np.isfinite(d).all() or np.any(d<0) or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0):raise ValueError('Invalid feature distances')
 manifest=groupdata.get('source_manifest_sha256',{})
 if set(manifest)!=set(paths) or any(morph.sha256_file(p)!=h for p,h in manifest.items()):raise ValueError('Source photos changed or manifest incomplete')
 group_by_source={}
 for group in groupdata['groups']:
  for photo in group['photos']:
   path=str(Path(photo).resolve())
   if path in group_by_source or path not in lookup:raise ValueError('Invalid visual group membership')
   group_by_source[path]=group
 if set(group_by_source)!=set(paths):raise ValueError('Incomplete visual group membership')
 all_eligible={str(Path(r['photo']).resolve()) for s in study['scenes'] for r in s['heldout']};edit_hashes={str((Path(a.edits)/Path(p).name).resolve()):morph.sha256_file(Path(a.edits)/Path(p).name) for p in all_eligible};cache={};weights={}
 def pair(path):
  if path not in cache:
   target=Path(a.edits)/Path(path).name
   with Image.open(path) as im: s1=ImageOps.exif_transpose(im).size
   with Image.open(target) as im: s2=ImageOps.exif_transpose(im).size
   if s1!=s2:raise ValueError('Original and edit dimensions differ: '+path)
   cache[path]=(morph.load(path,study['size']),morph.load(target,study['size']))
  return cache[path]
 def score(rep,photo):
  if rep not in weights:
   x,y=pair(rep);weights[rep]=poly_color.fit(x,y,regularization=study['regularization'])[0]
  x,y=pair(photo);return float(np.sqrt(np.mean((poly_color.render(x,weights[rep])-y)**2)))
 rows=[];excluded=[];choices=[]
 for s in study['scenes']:
  eligible={str(Path(r['photo']).resolve()) for r in s['heldout']};fixed=str(Path(s['heldout'][0]['photo']).resolve());scenes={group_by_source[p]['id']:group_by_source[p] for p in eligible}
  for gid,g in scenes.items():
   if g['source_scene']!=s['scene']:raise ValueError('Chronological scene membership mismatch')
   members=sorted(p for p in eligible if group_by_source[p]['id']==gid);medoid=str(Path(g['representative']).resolve());ref=min(members,key=lambda p:(d[lookup[medoid],lookup[p]],p));choices.append({'scene':s['scene'],'visual_group':gid,'visual_medoid':medoid,'eligible_reference':ref,'eligible_photos':len(members)})
   for photo in members:
    if photo in {fixed,ref}:excluded.append({'photo':photo,'group':gid,'reason':'one of the compared fitting references'});continue
    rows.append({'scene':s['scene'],'group':gid,'photo':photo,'fixed_reference':fixed,'visual_reference':ref,'fixed_rmse':score(fixed,photo),'visual_rmse':score(ref,photo)})
 if not rows:raise ValueError('No jointly held-out photos')
 if any(morph.sha256_file(p)!=h for p,h in edit_hashes.items()) or any(morph.sha256_file(p)!=h for p,h in manifest.items()):raise ValueError('Input changed during evaluation')
 b=np.array([r['fixed_rmse'] for r in rows]);v=np.array([r['visual_rmse'] for r in rows]);summary={'jointly_heldout_photos':len(rows),'scene_count':len(set(r['scene'] for r in rows)),'visual_groups_with_holdouts':len(set(r['group'] for r in rows)),'baseline_mean_rmse':float(b.mean()),'visual_mean_rmse':float(v.mean()),'visual_lower':int(np.sum(v<b-1e-8)),'visual_higher':int(np.sum(v>b+1e-8)),'same':int(np.sum(np.abs(v-b)<=1e-8))}
 out={'scope':'One-album exploratory comparison. Each visual subgroup uses one fixed eligible edited reference, chosen using distances to its visual medoid before looking at target edits. The chronological baseline uses one fixed prior reference per scene. A photo is evaluated only if it is neither reference. Both methods fit the same polynomial model and compare only this joint held-out set. Many photos/groups may be excluded; the result does not measure album coverage, segmentation, normalization, artistic quality, or generalization. Chronological groups, visual threshold, and source/edit alignment are not independently verified and edited intent may vary.','size':study['size'],'regularization':study['regularization'],'summary':summary,'choices':choices,'excluded':excluded,'photos':rows,'edit_sha256':edit_hashes}
 morph.protect_inputs([a.sensitivity,a.grouping,a.distances]+paths+list(edit_hashes),[a.out]);morph.protect_existing([a.out],False);morph.dump(out,a.out);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
