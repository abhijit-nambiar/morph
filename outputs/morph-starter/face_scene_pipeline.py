#!/usr/bin/env python3
"""Experimental scene-level face-light fitting and full-resolution replay."""
import argparse,json,math,re,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import morph,prompt_target,gamut_renderer
import face_core as face
import face_quality


def parse_prompt(prompt):
    clauses=[c.strip() for c in re.split(r'[,;]',re.sub(r'\band\b',',',prompt.lower().strip()))]
    if not clauses or any(not c for c in clauses):raise ValueError('Prompt must have nonempty clauses')
    local=None;global_parts=[]
    for clause in clauses:
        phrase=clause;intensity=1.
        for prefix,factor in (('slightly ',.5),('strongly ',1.5)):
            if phrase.startswith(prefix):intensity=factor;phrase=phrase[len(prefix):];break
        if phrase in ('brighten faces','darken faces'):
            if local is not None:raise ValueError('Only one face lighting instruction is supported')
            local=(.3 if phrase=='brighten faces' else -.3)*intensity
        else:global_parts.append(clause)
    if local is None:raise ValueError('Include brighten faces or darken faces')
    settings,interpreted=prompt_target.parse(', '.join(global_parts)) if global_parts else ({},[])
    return morph.soft_settings('neutral',settings),local,interpreted


def checked_faces(value,allow_empty=False):
    if not isinstance(value,list) or (not value and not allow_empty):raise ValueError('Expected a list of face boxes; representative requires at least one')
    result=[]
    for b in value:
        if not isinstance(b,dict):raise ValueError('Invalid face box')
        coords=[]
        for k in ('x','y','width','height'):
            v=b.get(k)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Invalid face box coordinate')
            coords.append(float(v))
        x,y,w,h=coords
        if x<0 or y<0 or w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001:raise ValueError('Face box must be within the image')
        result.append(dict(zip(('x','y','width','height'),coords)))
    return result


def fit_scene(a):
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));g=next((g for g in groups if g['id']==a.scene),None)
    if g is None:raise ValueError('Unknown scene')
    raw=json.loads(Path(a.detections).read_text());lookup={}
    for r in raw:
        path=str(Path(r['path']).resolve())
        if path in lookup:raise ValueError('Duplicate detection for '+path)
        lookup[path]=r
    manifest=[]
    for path in g['photos']:
        path=str(Path(path).resolve());d=lookup.get(path)
        if not d or d.get('error'):raise ValueError('Missing or failed face detection: '+path)
        checksum=morph.sha256_file(path);detection_hash=d.get('source_sha256')
        if detection_hash is None and not a.allow_unverified_detections:raise ValueError('Detection has no source checksum: '+path+'; regenerate it or explicitly use --allow-unverified-detections')
        if detection_hash is not None and detection_hash!=checksum:raise ValueError('Image changed since face detection: '+path)
        manifest.append({'path':path,'sha256':checksum,'faces':checked_faces(d.get('faces'),allow_empty=True),'detection_validation':'verified' if detection_hash is not None else 'unverified'})
    rep=str(Path(g['representative']).resolve());boxes=next(r['faces'] for r in manifest if r['path']==rep)
    if not boxes and any(r['faces'] for r in manifest):raise ValueError('Representative has no detected faces; choose a representative with faces')
    local_status='fitted' if boxes else 'unavailable: no faces detected in scene; review'
    x=morph.load(rep,a.size);target_path=None;target_sha256=None
    if a.targets:
        target_path=Path(a.targets)/Path(rep).name
        if not target_path.is_file():raise ValueError('Missing representative target: '+str(target_path))
        with Image.open(rep) as original,Image.open(target_path) as edited:
            if morph.oriented_size(original)!=morph.oriented_size(edited):raise ValueError('Representative target dimensions differ: '+str(target_path))
        target_sha256=morph.sha256_file(target_path);y=morph.load(target_path,a.size);settings=None;local=None;interpreted=None
    else:
        settings,local,interpreted=parse_prompt(a.prompt);y=face.target(x,boxes,settings,local)
    w=face.fit(x,y,boxes,a.regularization)
    out=Path(a.out);outputs=[out/'fit.json',out/'target-preview.jpg',out/'reconstruction-preview.jpg']
    morph.protect_inputs([a.groups,a.detections,*g['photos']]+([target_path] if target_path else []),outputs);morph.protect_existing(outputs,False)
    morph.save(y,out/'target-preview.jpg');morph.save(face.render(x,boxes,w),out/'reconstruction-preview.jpg')
    morph.dump({'version':1,'scope':'Experimental face-relative lighting. Boxes are not validated by this tool; masks extend outside faces.','scene':g['id'],'representative':rep,'prompt':a.prompt,'target_engine':'supplied' if target_path else 'synthetic-face','target_path':str(target_path.resolve()) if target_path else None,'target_sha256':target_sha256,'global_interpretation':interpreted,'target_settings':settings,'target_face_light':local,'local_status':local_status,'fit_size':a.size,'regularization':a.regularization,'weights':w.tolist(),'manifest':manifest},out/'fit.json')
    print('Saved scene fit for',len(manifest),'photos;',sum(not r['faces'] for r in manifest),'with no detected faces')


def validate_fit(path):
    fit=json.loads(Path(path).read_text())
    if fit.get('version')!=1:raise ValueError('Unsupported fit version')
    w=np.asarray(fit['weights'],dtype=np.float64)
    if w.shape!=(13,3) or not np.isfinite(w).all():raise ValueError('Invalid fitted weights')
    manifest=fit['manifest']
    if not isinstance(manifest,list) or not manifest:raise ValueError('Invalid manifest')
    seen=set();sources=[]
    for item in manifest:
        path=str(Path(item['path']).resolve())
        if path in seen:raise ValueError('Duplicate image in fit manifest')
        seen.add(path);boxes=checked_faces(item['faces'],allow_empty=True)
        if morph.sha256_file(path)!=item['sha256']:raise ValueError('Source changed since fit: '+path)
        sources.append((path,boxes))
    if len({Path(p).name for p,_ in sources})!=len(sources):raise ValueError('Duplicate output filenames')
    if str(Path(fit['representative']).resolve()) not in seen:raise ValueError('Representative missing from manifest')
    return fit,w,sources


def load_corrections(path):
    config=json.loads(Path(path).read_text()) if path else {}
    if not isinstance(config,dict) or set(config)-{'album','scenes','photos'}:raise ValueError('Corrections must contain only album, scenes, photos objects')
    album=config.get('album',{});scenes=config.get('scenes',{});photos=config.get('photos',{})
    if not all(isinstance(x,dict) for x in (album,scenes,photos)):raise ValueError('Correction sections must be objects')
    gamut_renderer.vec(album)
    return config


def validate_corrections(config,scene_sources):
    scenes=config.get('scenes',{});photos=config.get('photos',{});ids=set(scene_sources)
    if (set(scenes)|set(photos))-ids:raise ValueError('Unknown scene in corrections')
    for sid,sources in scene_sources.items():
        scene_corrections=scenes.get(sid,{});photo_corrections=photos.get(sid,{})
        if not isinstance(scene_corrections,dict) or not isinstance(photo_corrections,dict):raise ValueError('Invalid scene or photo corrections')
        gamut_renderer.vec(scene_corrections)
        if set(photo_corrections)-{Path(path).name for path,_ in sources}:raise ValueError('Unknown photo in corrections for '+sid)
        for correction in photo_corrections.values():
            if not isinstance(correction,dict):raise ValueError('Invalid photo correction')
            gamut_renderer.vec(correction)


def apply_scene(a):
    fit,w,sources=validate_fit(a.fit)
    config=getattr(a,'correction_config',None)
    if config is None:
        config=load_corrections(a.corrections);validate_corrections(config,{fit['scene']:sources})
    scene=fit['scene'];album_vector=gamut_renderer.vec(config.get('album',{}));scene_vector=gamut_renderer.vec(config.get('scenes',{}).get(scene,{}));photo_vectors={name:gamut_renderer.vec(values) for name,values in config.get('photos',{}).get(scene,{}).items()}
    preview_names=getattr(a,'preview_photo',None)
    if preview_names:
        if a.preview_samples:raise ValueError('Choose --preview-photo or --preview-samples, not both')
        lookup={Path(path).name:(path,boxes) for path,boxes in sources}
        if len(set(preview_names))!=len(preview_names) or set(preview_names)-set(lookup):raise ValueError('Preview photos must be unique scene filenames')
        sources=[lookup[name] for name in preview_names]
    elif a.preview_samples:
        rep=fit['representative'];sources.sort(key=lambda p:(p[0]!=rep,Path(p[0]).name not in photo_vectors,p[0]));sources=sources[:a.preview_samples]
    out=Path(a.out);outputs=[out/'report.json']+[out/'exports'/Path(p).name for p,_ in sources]
    morph.protect_inputs([a.fit]+([a.corrections] if getattr(a,'corrections',None) else [])+[p for p,_ in sources],outputs);morph.protect_existing(outputs,False)
    rows=[]
    for path,boxes in sources:
        x=morph.load(path);photo_vector=photo_vectors.get(Path(path).name)
        def render_tile(tile,weights,start,shape):
            if photo_vector is not None and np.any(photo_vector):tile=gamut_renderer.render(tile,photo_vector)
            tile=face.model_tile(tile,weights,boxes,start,shape)
            if np.any(scene_vector):tile=gamut_renderer.render(tile,scene_vector)
            if np.any(album_vector):tile=gamut_renderer.render(tile,album_vector)
            return tile
        def renderer(tile,weights):return render_tile(tile,weights,0,tile.shape)
        renderer.render_tile=render_tile
        dest=out/'exports'/Path(path).name;morph.save_render(x,w,dest,renderer=renderer,source=path)
        sample=morph.load(path,512);diagnostic=face_quality.diagnose(sample,w,boxes,config.get('album',{}),config.get('scenes',{}).get(scene,{}),config.get('photos',{}).get(scene,{}).get(Path(path).name,{}))
        rows.append({'source':path,'output':str(dest),'clipping':diagnostic,'faces':len(boxes),'local_lighting':'applied around detected faces' if boxes else 'skipped: no faces detected; inspect photo','dimensions':[x.shape[1],x.shape[0]]});print(Path(path).name,'(no faces detected; review)' if not boxes else '',flush=True)
    morph.dump({'scope':'Experimental face-relative reconstruction. Face boxes are unverified and local lighting affects surroundings.','scene':fit['scene'],'source_fit':str(Path(a.fit)),'local_status':fit.get('local_status','legacy-unknown'),'mode':'preview' if a.preview_samples or preview_names else 'scene','correction_renderer':'gamut','corrections':{'album':config.get('album',{}),'scene':config.get('scenes',{}).get(scene,{}),'photos':config.get('photos',{}).get(scene,{})},'operation_order':['photo correction','fitted look','scene correction','album correction'],'photos_without_detected_faces':sum(not r['faces'] for r in rows),'near_limit_flags':sum(r['clipping']['near_limit_flag'] for r in rows),'diagnostic_scope':'Reduced-resolution, pre-JPEG estimates; not proof of visible damage or quality.','photos':rows},out/'report.json')


def fit_album(a):
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));out=Path(a.out)
    if out.exists():raise ValueError('Output directory already exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.face-fit-',dir=out.parent) as temp:
        staged=Path(temp)/'album';staged.mkdir()
        for g in groups:
            args=argparse.Namespace(groups=a.groups,detections=a.detections,scene=g['id'],prompt=a.prompt,targets=a.targets,out=str(staged/g['id']),size=a.size,regularization=a.regularization,allow_unverified_detections=a.allow_unverified_detections)
            fit_scene(args)
        morph.dump({'version':1,'scenes':[g['id'] for g in groups],'prompt':a.prompt,'target_engine':'supplied' if a.targets else 'synthetic-face'},staged/'index.json')
        staged.rename(out)
    print('Saved',len(groups),'scene fits')


def apply_album(a):
    fits=Path(a.fits);index=json.loads((fits/'index.json').read_text());scenes=index.get('scenes')
    if index.get('version')!=1 or not isinstance(scenes,list) or not scenes:raise ValueError('Invalid album fit index')
    if len(set(scenes))!=len(scenes) or any(not isinstance(s,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',s) for s in scenes):raise ValueError('Invalid or duplicate scene ID')
    out=Path(a.out)
    if out.exists():raise ValueError('Output directory already exists')
    scene_sources={}
    for scene in scenes:
        fit,_,sources=validate_fit(fits/scene/'fit.json')
        if fit['scene']!=scene:raise ValueError('Fit scene does not match album index')
        scene_sources[scene]=sources
    config=load_corrections(a.corrections);validate_corrections(config,scene_sources)
    out.mkdir(parents=True)
    for scene in scenes:apply_scene(argparse.Namespace(fit=str(fits/scene/'fit.json'),out=str(out/scene),preview_samples=a.preview_samples,corrections=a.corrections,correction_config=config))
    morph.dump({'scope':'Experimental face-relative album reconstruction; inspect detections and output.','mode':'preview' if a.preview_samples else 'album','scenes':scenes,'source_fits':str(fits)},out/'report.json')
    print('Rendered',len(scenes),'scenes')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit');f.add_argument('groups');f.add_argument('detections');f.add_argument('--scene',required=True);ft=f.add_mutually_exclusive_group(required=True);ft.add_argument('--prompt');ft.add_argument('--targets',help='Folder of same-name edited representatives');f.add_argument('--out',required=True);f.add_argument('--size',type=int,default=320);f.add_argument('--regularization',type=float,default=.0001);f.add_argument('--allow-unverified-detections',action='store_true',help='Use legacy boxes that have no source checksum')
    r=sub.add_parser('apply');r.add_argument('fit');r.add_argument('--out',required=True);r.add_argument('--corrections',help='JSON album, scene, and photo corrections');r.add_argument('--preview-samples',type=int,default=0);r.add_argument('--preview-photo',action='append',help='Render a specific scene filename; may be repeated')
    fa=sub.add_parser('fit-album');fa.add_argument('groups');fa.add_argument('detections');fat=fa.add_mutually_exclusive_group(required=True);fat.add_argument('--prompt');fat.add_argument('--targets',help='Folder of same-name edited representatives');fa.add_argument('--out',required=True);fa.add_argument('--size',type=int,default=320);fa.add_argument('--regularization',type=float,default=.0001);fa.add_argument('--allow-unverified-detections',action='store_true')
    ra=sub.add_parser('apply-album');ra.add_argument('fits');ra.add_argument('--out',required=True);ra.add_argument('--corrections',help='JSON album, scene, and photo corrections');ra.add_argument('--preview-samples',type=int,default=0)
    a=p.parse_args()
    if a.command in ('fit','fit-album'):
        if a.size<32 or not math.isfinite(a.regularization) or a.regularization<0:raise ValueError('Invalid size or regularization')
        fit_scene(a) if a.command=='fit' else fit_album(a)
    else:
        if a.preview_samples<0:raise ValueError('preview samples must be nonnegative')
        apply_scene(a) if a.command=='apply' else apply_album(a)
if __name__=='__main__':main()
