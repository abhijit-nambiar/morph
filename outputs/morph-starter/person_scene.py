#!/usr/bin/env python3
"""Experimental mask-aware scene fit and full-resolution replay."""
import argparse,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image
import morph,poly_color,prompt_target,gamut_renderer,face_scene_pipeline


def parse_prompt(prompt):
    clauses=[v.strip() for v in re.split(r'[,;]',re.sub(r'\band\b',',',prompt.lower()))]
    if not clauses or any(not c for c in clauses):raise ValueError('Empty prompt clause')
    local=None;remaining=[]
    for clause in clauses:
        match=re.fullmatch(r'(?:(slightly|strongly) )?(brighten|darken) people',clause)
        if match:
            if local is not None:raise ValueError('Only one people-light instruction is allowed')
            local=(.24 if match.group(2)=='brighten' else -.24)*{'slightly':.5,'strongly':1.5,None:1}[match.group(1)]
        else:remaining.append(clause)
    if local is None:raise ValueError('Include brighten people or darken people')
    settings,interpreted=prompt_target.parse(', '.join(remaining)) if remaining else ({},[])
    return settings,local,interpreted


def mask_array(path,shape):
    with Image.open(path) as im:
        if im.format!='PNG' or im.mode not in ('L','I;16'):raise ValueError('Mask must be a grayscale PNG: '+str(path))
        im=im.convert('L').resize((shape[1],shape[0]),Image.Resampling.BILINEAR)
        return np.asarray(im,dtype=np.float32)/255


def features(x,m):return np.concatenate([poly_color.features(x),m[...,None]*x*(1-x),m[...,None]],axis=-1)

def fit_mapping(x,y,m,reg):
    f=features(x,m).reshape(-1,14).astype(np.float64);b=(y-x).reshape(-1,3).astype(np.float64);idx=np.random.default_rng(0).choice(len(f),min(len(f),30000),replace=False);a=f[idx];b=b[idx]
    return np.linalg.lstsq(a.T@a/len(a)+reg*np.eye(14),a.T@b/len(a),rcond=None)[0]

def fit_scene(a):
    if not 64<=a.size<=1024 or not math.isfinite(a.regularization) or not 0<a.regularization<=.1:raise ValueError('Invalid fit size or regularization')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));group=next((g for g in groups if g['id']==a.scene),None)
    if group is None:raise ValueError('Unknown scene')
    entries=json.loads(Path(a.masks).read_text());lookup={}
    for row in entries:
        path=str(Path(row['source']).resolve())
        if path in lookup:raise ValueError('Duplicate mask entry')
        mask=str(Path(row['mask']).resolve())
        if morph.sha256_file(path)!=row['source_sha256'] or morph.sha256_file(mask)!=row['mask_sha256']:raise ValueError('Photo or mask changed since mask manifest')
        lookup[path]={'source':path,'source_sha256':row['source_sha256'],'mask':mask,'mask_sha256':row['mask_sha256']}
    sources=[lookup[str(Path(p).resolve())] for p in group['photos']];rep=str(Path(group['representative']).resolve());settings,local,interpreted=parse_prompt(a.prompt)
    for row in sources:
        with Image.open(row['mask']) as im:
            if im.format!='PNG' or im.mode not in ('L','I;16'):raise ValueError('Mask must be grayscale PNG')
    x=morph.load(rep,a.size);m=mask_array(next(row['mask'] for row in sources if row['source']==rep),x.shape)
    if float(m.mean())<.001:raise ValueError('Representative has no usable person mask; inspect masks and choose another representative')
    y=morph.pillow_soft_target(x,'neutral',settings);y=np.clip(y+m[...,None]*local*y*(1-y),0,1);w=fit_mapping(x,y,m,a.regularization)
    out=Path(a.out);outputs=[out/'fit.json',out/'target-preview.jpg',out/'reconstruction-preview.jpg'];morph.protect_inputs([a.groups,a.masks]+[r['source'] for r in sources]+[r['mask'] for r in sources],outputs);morph.protect_existing(outputs,False);out.mkdir(parents=True,exist_ok=True)
    morph.save(y,outputs[1]);morph.save(np.clip(x+features(x,m)@w,0,1),outputs[2]);morph.dump({'version':1,'kind':'experimental-person-scene-v1','scope':'Soft predicted person masks can miss people or include background. Inspect masks and full-resolution previews. Synthetic targets and reconstruction share masks.','scene':group['id'],'representative':rep,'prompt':a.prompt,'global_interpretation':interpreted,'target_settings':settings,'target_person_light':local,'fit_size':a.size,'regularization':a.regularization,'weights':w.tolist(),'manifest':sources},outputs[0]);print('Fitted',group['id'])


def apply_scene(a):
    fit=json.loads(Path(a.fit).read_text());w=np.asarray(fit['weights'],dtype=np.float64)
    if fit.get('kind')!='experimental-person-scene-v1' or w.shape!=(14,3) or not np.isfinite(w).all():raise ValueError('Invalid fit')
    rows=fit['manifest'];paths=set();names=set()
    for row in rows:
        path=str(Path(row['source']).resolve());mask=str(Path(row['mask']).resolve());name=Path(path).name.lower()
        if path in paths or name in names:raise ValueError('Duplicate source or output name')
        paths.add(path);names.add(name)
        if morph.sha256_file(path)!=row['source_sha256'] or morph.sha256_file(mask)!=row['mask_sha256']:raise ValueError('Source or mask changed since fit')
    config=face_scene_pipeline.load_corrections(a.corrections);scene=fit['scene']
    face_scene_pipeline.validate_corrections(config,{scene:[(r['source'],[]) for r in fit['manifest']]})
    album_vector=gamut_renderer.vec(config.get('album',{}));scene_vector=gamut_renderer.vec(config.get('scenes',{}).get(scene,{}));photo_vectors={name:gamut_renderer.vec(value) for name,value in config.get('photos',{}).get(scene,{}).items()}
    if a.preview_photo:
        wanted=set(a.preview_photo)
        if len(wanted)!=len(a.preview_photo) or wanted-{Path(r['source']).name for r in rows}:raise ValueError('Unknown or duplicate preview photo')
        rows=[r for r in rows if Path(r['source']).name in wanted]
    out=Path(a.out);outputs=[out/'report.json']+[out/'exports'/Path(r['source']).name for r in rows];morph.protect_inputs([a.fit]+([a.corrections] if a.corrections else [])+[r['source'] for r in fit['manifest']]+[r['mask'] for r in fit['manifest']],outputs);morph.protect_existing(outputs,False);results=[]
    for row in rows:
        x=morph.load(row['source']);m=mask_array(row['mask'],x.shape);photo_vector=photo_vectors.get(Path(row['source']).name)
        class Renderer:
            def render_tile(self,tile,weights,start,shape):
                if photo_vector is not None and np.any(photo_vector):tile=gamut_renderer.render(tile,photo_vector)
                tile=np.clip(tile+features(tile,m[start:start+tile.shape[0]])@weights,0,1)
                if np.any(scene_vector):tile=gamut_renderer.render(tile,scene_vector)
                if np.any(album_vector):tile=gamut_renderer.render(tile,album_vector)
                return tile
        dest=out/'exports'/Path(row['source']).name;morph.save_render(x,w,dest,renderer=Renderer(),source=row['source']);results.append({'source':row['source'],'output':str(dest.resolve()),'mask_mean':float(m.mean()),'review_flag':float(m.mean())<.001})
    morph.dump({'scene':fit['scene'],'source_fit':str(Path(a.fit).resolve()),'mode':'preview' if a.preview_photo else 'scene','scope':'Inspect mask edges and full-resolution outputs. Empty masks receive global components only; corrections are manual, not automatic normalization.','corrections':{'album':config.get('album',{}),'scene':config.get('scenes',{}).get(scene,{}),'photos':config.get('photos',{}).get(scene,{})},'operation_order':['photo correction','fitted look','scene correction','album correction'],'photos':results},out/'report.json');print('Rendered',len(results),'photos')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True);f=sub.add_parser('fit');f.add_argument('groups');f.add_argument('masks');f.add_argument('--scene',required=True);f.add_argument('--prompt',required=True);f.add_argument('--size',type=int,default=256);f.add_argument('--regularization',type=float,default=.00001);f.add_argument('--out',required=True);a=sub.add_parser('apply');a.add_argument('fit');a.add_argument('--preview-photo',action='append');a.add_argument('--corrections');a.add_argument('--out',required=True);args=p.parse_args();fit_scene(args) if args.cmd=='fit' else apply_scene(args)
if __name__=='__main__':main()
