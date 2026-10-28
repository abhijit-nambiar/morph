#!/usr/bin/env python3
"""Experimental multi-scene person-aware fitting and full-resolution album replay."""
import argparse,json,os,re,tempfile
from pathlib import Path
import numpy as np
import morph,person_scene,face_scene_pipeline


def output_dir(value):
 out=Path(value).resolve()
 if out.exists() or not out.parent.is_dir():raise ValueError('Output already exists or parent is missing')
 return out

def fit_album(a):
 groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));out=output_dir(a.out)
 with tempfile.TemporaryDirectory(prefix='.person-fits-',dir=out.parent) as td:
  staged=Path(td)/'fits';staged.mkdir()
  for g in groups:
   person_scene.fit_scene(argparse.Namespace(groups=a.groups,masks=a.masks,scene=g['id'],prompt=a.prompt,size=a.size,regularization=a.regularization,out=str(staged/g['id'])))
  morph.dump({'version':1,'kind':'experimental-person-album-v1','scenes':[g['id'] for g in groups],'prompt':a.prompt},staged/'index.json');os.rename(staged,out)
 print('Fitted',len(groups),'scenes')

def apply_album(a):
 out=output_dir(a.out);fits=Path(a.fits).resolve();index=json.loads((fits/'index.json').read_text());scenes=index.get('scenes')
 if index.get('kind')!='experimental-person-album-v1' or not isinstance(scenes,list) or not scenes or len(set(scenes))!=len(scenes) or any(not isinstance(s,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',s) for s in scenes):raise ValueError('Invalid album index')
 if a.preview_samples<0:raise ValueError('Preview samples must be non-negative')
 scene_sources={};fit_data={};seen=set()
 for scene in scenes:
  path=fits/scene/'fit.json';fit=json.loads(path.read_text());w=np.asarray(fit['weights'],dtype=float)
  if fit.get('kind')!='experimental-person-scene-v1' or fit.get('scene')!=scene or w.shape!=(14,3) or not np.isfinite(w).all() or not fit.get('manifest'):raise ValueError('Invalid scene fit')
  sources=[];names=set()
  for row in fit['manifest']:
   source=str(Path(row['source']).resolve());mask=str(Path(row['mask']).resolve());name=Path(source).name.lower()
   if source in seen or name in names:raise ValueError('Duplicate photo or filename')
   seen.add(source);names.add(name)
   if morph.sha256_file(source)!=row['source_sha256'] or morph.sha256_file(mask)!=row['mask_sha256']:raise ValueError('Source or mask changed since fit')
   sources.append((source,[]))
  if str(Path(fit['representative']).resolve()) not in {p for p,_ in sources}:raise ValueError('Representative is missing')
  scene_sources[scene]=sources;fit_data[scene]=fit
 config=face_scene_pipeline.load_corrections(a.corrections);face_scene_pipeline.validate_corrections(config,scene_sources)
 with tempfile.TemporaryDirectory(prefix='.person-album-',dir=out.parent) as td:
  staged=Path(td)/'album';staged.mkdir();reports=[]
  for scene in scenes:
   fit=fit_data[scene];scene_config={'album':config.get('album',{}),'scenes':{scene:config.get('scenes',{}).get(scene,{})},'photos':{scene:config.get('photos',{}).get(scene,{})}};config_path=Path(td)/'corrections.json';morph.dump(scene_config,config_path)
   names=[Path(r['source']).name for r in fit['manifest']];rep=Path(fit['representative']).name;selected=([rep]+[n for n in names if n!=rep])[:a.preview_samples] if a.preview_samples else None
   person_scene.apply_scene(argparse.Namespace(fit=str(fits/scene/'fit.json'),out=str(staged/scene),preview_photo=selected,corrections=str(config_path)))
   report_path=staged/scene/'report.json';report=json.loads(report_path.read_text())
   for row in report['photos']:row['output']=str(out/scene/'exports'/Path(row['output']).name)
   morph.dump(report,report_path);reports.append({'scene':scene,'photos':len(report['photos']),'report':scene+'/report.json'})
  morph.dump({'kind':'experimental-person-album-output-v1','scope':'One shared mask-aware mapping per scene; corrections are manual. Inspect masks and full-resolution output.','mode':'preview' if a.preview_samples else 'album','fits':str(fits),'scenes':reports,'photos':sum(r['photos'] for r in reports)},staged/'report.json');os.rename(staged,out)
 print('Rendered',len(scenes),'scenes')

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True);f=sub.add_parser('fit');f.add_argument('groups');f.add_argument('masks');f.add_argument('--prompt',required=True);f.add_argument('--size',type=int,default=256);f.add_argument('--regularization',type=float,default=.00001);f.add_argument('--out',required=True);a=sub.add_parser('apply');a.add_argument('fits');a.add_argument('--preview-samples',type=int,default=0);a.add_argument('--corrections');a.add_argument('--out',required=True);args=p.parse_args();fit_album(args) if args.cmd=='fit' else apply_album(args)
if __name__=='__main__':main()
