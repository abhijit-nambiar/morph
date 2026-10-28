#!/usr/bin/env python3
"""Deterministic album-edit reconstruction experiment. Python 3, NumPy, Pillow."""
import argparse, json, math, os, time, re, hashlib, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageCms, ImageEnhance, ImageDraw

NAMES=['exposure','temperature','tint','contrast','shadows','highlights','saturation']
BOUNDS=np.array([[-1.5,1.5],[-.4,.4],[-.3,.3],[-.6,.6],[-.5,.5],[-.5,.5],[-.8,.8]])
PRESETS={'warm':{'exposure':.16,'temperature':.13,'contrast':.15,'shadows':.12,'highlights':-.13,'saturation':.10},'cool':{'exposure':.1,'temperature':-.13,'contrast':.12,'shadows':.10,'highlights':-.10,'saturation':-.08},'soft':{'exposure':.12,'temperature':.05,'contrast':-.16,'shadows':.16,'highlights':-.16,'saturation':-.09},'neutral':{}}

def vec(d):
    if any(k not in NAMES for k in d): raise ValueError('Unknown controls: '+str(set(d)-set(NAMES)))
    v=np.array([float(d.get(k,0)) for k in NAMES]);
    if not np.isfinite(v).all() or np.any(v<BOUNDS[:,0]) or np.any(v>BOUNDS[:,1]): raise ValueError('Recipe outside bounds or nonfinite')
    return v

def recipe(v): return {k:float(x) for k,x in zip(NAMES,v)}
def load(path,maxside=None):
    with Image.open(path) as im:
        if maxside:
            im.draft('RGB', (maxside, maxside))
        im=ImageOps.exif_transpose(im)
        if maxside: im.thumbnail((maxside,maxside),Image.Resampling.LANCZOS)
        if im.info.get('icc_profile'):
            import io
            try: im=ImageCms.profileToProfile(im,ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile'])),ImageCms.createProfile('sRGB'),outputMode='RGB')
            except Exception as e: raise ValueError(f'Cannot convert ICC profile for {path}: {e}')
        else: im=im.convert('RGB')
        return np.asarray(im,dtype=np.float32)/255

def render(x,p):
    # Smooth per-pixel, ordered global light/color operations; no spatial movement.
    e,t,tint,c,sh,hi,s=p
    y=x*np.float32(2**e)*np.array([1+t,1+tint,1-t],dtype=np.float32)
    lum=np.sum(y*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True)
    y=y+(lum*(1-np.clip(lum,0,1)))*(sh*(1-np.clip(lum,0,1))+hi*np.clip(lum,0,1))*2
    y=(y-.5)*(1+c)+.5
    lum=np.sum(y*np.array([.2126,.7152,.0722],dtype=np.float32),axis=-1,keepdims=True)
    return np.clip(lum+(y-lum)*(1+s),0,1)

def pillow_target(x,name):
    """Independent synthetic editor, useful for cross-renderer reconstruction."""
    settings={'warm':(1.08,1.12,1.10,(1.07,1.00,.94)), 'cool':(1.05,1.10,.93,(.95,1.00,1.07)), 'soft':(1.07,.88,.90,(1.02,1.00,.98)), 'neutral':(1.,1.,1.,(1.,1.,1.))}
    bright,contrast,color,gains=settings[name]
    im=Image.fromarray(np.uint8(np.clip(x*255+.5,0,255)))
    im=ImageEnhance.Brightness(im).enhance(bright)
    im=ImageEnhance.Contrast(im).enhance(contrast)
    im=ImageEnhance.Color(im).enhance(color)
    return np.clip(np.asarray(im,dtype=np.float32)/255*np.array(gains,dtype=np.float32),0,1)

def pillow_selective_target(x,name):
    """Separate target pipeline: Pillow global controls plus HSV range edits."""
    base=pillow_target(x,name)
    controls={'warm':(-.23,.20),'cool':(.15,-.16),'soft':(-.16,-.08),'neutral':(0.,0.)}
    cool,warm=controls[name]
    hsv=np.asarray(Image.fromarray(np.uint8(base*255+.5)).convert('HSV'),dtype=np.float32)/255
    hue=hsv[...,0];cool_weight=np.clip(1-np.abs(hue-.62)/.18,0,1);warm_distance=np.minimum(np.abs(hue-.08),1-np.abs(hue-.08));warm_weight=np.clip(1-warm_distance/.14,0,1)
    hsv[...,1]=np.clip(hsv[...,1]*(1+cool*cool_weight+warm*warm_weight),0,1)
    packed=np.uint8(np.clip(hsv*255+.5,0,255))
    return np.asarray(Image.frombytes('HSV',(packed.shape[1],packed.shape[0]),packed.tobytes()).convert('RGB'),dtype=np.float32)/255

SOFT_NAMES=['tone_exposure','tone_contrast','tone_shadows','tone_highlights','tone_temperature','cool_saturation','warm_saturation']
SOFT_PRESETS={'warm':(.12,.12,.10,-.10,.05,-.15,-.05),'cool':(.08,.10,.08,-.12,-.05,-.04,-.12),'soft':(.09,-.12,.16,-.16,.02,-.12,-.08),'neutral':(0.,0.,0.,0.,0.,0.,0.)}
def soft_settings(name,custom=None):
    if custom is None:return dict(zip(SOFT_NAMES,SOFT_PRESETS[name]))
    if not isinstance(custom,dict) or any(k not in SOFT_NAMES for k in custom):raise ValueError('Unknown or invalid pillow-soft target controls')
    values={k:float(custom.get(k,0.)) for k in SOFT_NAMES}
    if not all(math.isfinite(x) and abs(x)<=.5 for x in values.values()):raise ValueError('pillow-soft target controls must be finite and between -0.5 and 0.5')
    return values

def pillow_soft_target(x,name,custom=None):
    """Independent bounded tone LUT plus HSV hue-range saturation."""
    settings=soft_settings(name,custom)
    exposure,contrast,shadows,highlights,temp,cool,warm=(settings[k] for k in SOFT_NAMES)
    image=Image.fromarray(np.uint8(np.clip(x*255+.5,0,255)));channels=image.split();values=np.arange(256,dtype=np.float64)/255
    curves=[]
    for index,channel in enumerate(channels):
        balance=temp if index==0 else (-temp if index==2 else 0.)
        y=values+(exposure+balance)*values*(1-values)+contrast*(values-.5)*values*(1-values)+shadows*values*(1-values)**2+highlights*values**2*(1-values)
        if np.any(y<0) or np.any(y>1) or np.any(np.diff(y)<0):raise ValueError('pillow-soft target tone curve is not monotonic and bounded')
        curves.append(channel.point(np.uint8(np.clip(y*255+.5,0,255)).tolist()))
    base=Image.merge('RGB',curves);hsv=np.asarray(base.convert('HSV'),dtype=np.float32)/255;h=hsv[...,0]
    cw=np.clip(1-np.abs(h-.62)/.18,0,1);distance=np.minimum(np.abs(h-.08),1-np.abs(h-.08));ww=np.clip(1-distance/.14,0,1)
    hsv[...,1]=np.clip(hsv[...,1]*(1+cool*cw+warm*ww),0,1);packed=np.uint8(np.clip(hsv*255+.5,0,255))
    return np.asarray(Image.frombytes('HSV',(packed.shape[1],packed.shape[0]),packed.tobytes()).convert('RGB'),dtype=np.float32)/255

def save(x,path,quality=95):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(np.clip(x*255+.5,0,255))).save(path,quality=quality,subsampling=0,icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())

def save_render(x,p,path,renderer=None,source=None):
    # Bound temporary arrays for high-resolution JPEG exports.
    renderer=renderer or render
    result=np.empty(x.shape,dtype=np.uint8)
    for start in range(0,x.shape[0],256):
        sl=slice(start,start+256)
        tile_render=getattr(renderer,'render_tile',None)
        value=tile_render(x[sl],p,start,x.shape) if tile_render else renderer(x[sl],p)
        result[sl]=np.uint8(value*255+.5)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    options={'quality':95,'subsampling':0,'icc_profile':ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()}
    if source is not None:
        with Image.open(source) as im:
            exif=im.getexif()
            if exif:
                if 274 in exif:exif[274]=1
                if 34665 in exif:
                    sub=exif.get_ifd(34665)
                    if 40961 in sub:sub[40961]=1
                    if 40962 in sub:sub[40962]=x.shape[1]
                    if 40963 in sub:sub[40963]=x.shape[0]
                options['exif']=exif.tobytes()
    Image.fromarray(result).save(path,**options)

def clipping_diagnostic(x,p,renderer=None):
    # Regular spatial sample of each source, comparing rendered and original
    # before JPEG encoding. Counts pixels with at least one near-limit channel.
    renderer=renderer or render
    stride=max(1,max(x.shape[:2])//512); sample=x[::stride,::stride]; edited=renderer(sample,p)
    def stats(im):
        return {'high':float(np.mean(np.any(im>=254/255,axis=-1))), 'low':float(np.mean(np.all(im<=1/255,axis=-1)))}
    before=stats(sample); after=stats(edited)
    high_increase=after['high']-before['high']; low_increase=after['low']-before['low']
    return {'sampled_pixels':int(sample.shape[0]*sample.shape[1]),'before':before,'after':after,'high_increase':high_increase,'low_increase':low_increase,'flag':high_increase>.01 or low_increase>.01}

def fit(a,b,iterations=45,seed=0):
    if iterations<1: raise ValueError('iterations must be positive')
    if a.shape!=b.shape: raise ValueError('Images have different dimensions; align/crop externally before fitting.')
    a=a.reshape(-1,3);b=b.reshape(-1,3)
    rng=np.random.default_rng(seed); idx=rng.choice(len(a),min(len(a),12000),replace=False); a=a[idx];b=b[idx]
    # Coordinate pattern search with step reduction, bounded, pixel reconstruction objective.
    p=np.zeros(len(NAMES)); steps=np.array([.18,.07,.06,.1,.09,.09,.12]); lo,hi=BOUNDS.T
    def loss(z):
        d=render(a,z)-b
        return float(np.mean(d*d)+1e-7*np.sum(z*z))
    best=loss(p); initial=best; calls=1
    for _ in range(iterations):
        improved=False
        for j in range(len(p)):
            cand=[]
            for direction in (-1,1):
                q=p.copy(); q[j]=np.clip(q[j]+direction*steps[j],lo[j],hi[j]); val=loss(q); calls+=1; cand.append((val,q))
            val,q=min(cand,key=lambda t:t[0])
            if val+1e-12<best: p=q;best=val;improved=True
        if not improved: steps*=.5
        if max(steps)<.0005: break
    return p,{'initial_mse':initial,'fit_mse':best,'rmse':math.sqrt(best),'evaluations':calls}

def run_fit(a,b,iterations=45,method='pattern'):
    if method=='pattern': return fit(a,b,iterations=iterations)
    if method=='gauss-newton':
        import solver
        return solver.fit_gauss_newton(a,b,iterations=iterations)
    raise ValueError('Unknown solver: '+str(method))

def features(x):
    # Features describe illumination, color and coarse composition, not semantic identity.
    y=x@np.array([.2126,.7152,.0722]); return np.array([np.mean(y),np.std(y),*np.quantile(y,[.1,.5,.9]),*x.mean((0,1)),*x.std((0,1))],dtype=float)

def cluster(paths,k):
    f=np.array([features(load(p,160)) for p in paths]); scale=np.std(f,axis=0); scale=np.maximum(scale,.04); z=(f-np.mean(f,axis=0))/scale
    k=min(k,len(paths)); centers=[int(np.argmin(np.linalg.norm(z-z.mean(0),axis=1)))]
    while len(centers)<k:
        dist=np.min(np.stack([np.sum((z-z[i])**2,axis=1) for i in centers]),axis=0); dist[centers]=-1; centers.append(int(np.argmax(dist)))
    c=z[centers].copy(); labels=np.zeros(len(paths),int)
    for _ in range(30):
        labels=np.argmin(((z[:,None,:]-c[None,:,:])**2).sum(axis=2),axis=1); new=c.copy()
        for i in range(k):
            if np.any(labels==i): new[i]=z[labels==i].mean(axis=0)
        if np.allclose(new,c): break
        c=new
    groups=[]
    for i in range(k):
        members=np.where(labels==i)[0]
        if not len(members): continue
        rep=members[np.argmin(((z[members]-c[i])**2).sum(axis=1))]
        groups.append({'id':f'scene-{i+1:02d}','representative':str(paths[rep]),'photos':[str(paths[j]) for j in members]})
    return groups

def chronological_groups(paths,gap_minutes=5):
    if gap_minutes<=0: raise ValueError('gap-minutes must be positive')
    timed=[];missing=[]
    for path in paths:
        with Image.open(path) as image:
            exif=image.getexif(); fields=exif.get_ifd(34665)
            stamp=fields.get(36867) or exif.get(306); camera=exif.get(272) or ''
            try: captured=datetime.strptime(stamp,'%Y:%m:%d %H:%M:%S')
            except (TypeError,ValueError): missing.append(path);continue
            timed.append((captured,camera,path))
    timed.sort(key=lambda row:(row[0],str(row[2]))); batches=[]
    for i,(captured,camera,path) in enumerate(timed):
        if i==0 or (captured-timed[i-1][0]).total_seconds()>gap_minutes*60 or camera!=timed[i-1][1]: batches.append([])
        batches[-1].append(path)
    batches.extend([[path] for path in missing])
    groups=[]
    for i,batch in enumerate(batches):
        f=np.array([features(load(p,160)) for p in batch]); scale=np.maximum(np.std(f,axis=0),.04); z=(f-f.mean(axis=0))/scale
        rep=int(np.argmin(np.sum(z*z,axis=1)))
        groups.append({'id':f'scene-{i+1:02d}','representative':str(batch[rep]),'photos':[str(p) for p in batch]})
    return groups

def hybrid_groups(paths,gap_minutes=5,approx_photos_per_group=8):
    if approx_photos_per_group<2:raise ValueError('approximate photos per group must be at least 2')
    time_groups=chronological_groups(paths,gap_minutes);split=[]
    for group in time_groups:
        members=[Path(p) for p in group['photos']]
        k=max(1,math.ceil(len(members)/approx_photos_per_group))
        pieces=cluster(members,k) if k>1 else [group]
        split.extend(pieces)
    for i,group in enumerate(split):group['id']=f'scene-{i+1:02d}'
    return split

def preview_samples(group,count):
    """Choose a representative and visually diverse members; no semantic labels."""
    if count<1:return []
    paths=group['photos'];images=[features(load(p,160)) for p in paths];f=np.array(images);scale=np.maximum(np.std(f,axis=0),.04);z=(f-f.mean(axis=0))/scale
    chosen=[paths.index(group['representative'])]
    while len(chosen)<min(count,len(paths)):
        dist=np.min(np.stack([np.sum((z-z[i])**2,axis=1) for i in chosen]),axis=0);dist[chosen]=-1;chosen.append(int(np.argmax(dist)))
    return [paths[i] for i in chosen]

def files(folder):
    p=Path(folder); return sorted((x.resolve() for x in p.iterdir() if x.is_file() and x.suffix.lower() in ('.jpg','.jpeg','.png')),key=lambda x:x.name)
def validate_groups(data):
    groups=data.get('groups')
    if not isinstance(groups,list) or not groups: raise ValueError('groups must be a nonempty list')
    ids=set(); seen=set()
    for group in groups:
        scene=group.get('id',''); photos=group.get('photos'); rep=group.get('representative')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',scene): raise ValueError('Invalid group ID: '+str(scene))
        if scene in ids: raise ValueError('Duplicate group ID: '+scene)
        ids.add(scene)
        if not isinstance(photos,list) or not photos or rep not in photos: raise ValueError('Every group needs photos and a representative contained in photos: '+scene)
        filenames=set()
        for photo in photos:
            path=Path(photo).resolve()
            if not path.is_file(): raise ValueError('Photo not found: '+str(path))
            if path in seen: raise ValueError('Photo appears in multiple assignments: '+str(path))
            if path.name.lower() in filenames: raise ValueError('Conflicting export filename in '+scene+': '+path.name)
            seen.add(path); filenames.add(path.name.lower())
    return groups
def dump(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')
def protect_inputs(inputs,outputs):
    protected={Path(p).resolve() for p in inputs}
    for path in outputs:
        if Path(path).resolve() in protected:raise ValueError('Output would overwrite an input: '+str(path))
def protect_existing(outputs,overwrite=False):
    if not overwrite:
        for path in outputs:
            if Path(path).exists():raise ValueError('Output already exists; choose a new output or pass --overwrite: '+str(path))
def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def oriented_size(image):
    w,h=image.size
    return (h,w) if image.getexif().get(274) in (5,6,7,8) else (w,h)
def comparison_sheet(rows,out,labels='original / human edit / reconstruction'):
    width,height=210,175
    pages=[]
    for first in range(0,len(rows),12):
        subset=rows[first:first+12]
        sheet=Image.new('RGB',(width*3,height*len(subset)),(248,248,248));d=ImageDraw.Draw(sheet)
        for i,(name,images) in enumerate(subset):
            y=i*height
            d.text((5,y+2),name+'  |  '+labels,fill=(20,20,20))
            for j,x in enumerate(images):
                im=Image.fromarray(np.uint8(np.clip(x*255+.5,0,255)));im.thumbnail((width-8,height-28))
                sheet.paste(im,(j*width+4+(width-8-im.width)//2,y+20+(height-28-im.height)//2))
        path=out/f'comparison-{first//12+1:02d}.jpg';sheet.save(path,quality=90);pages.append(path.name)
    return pages
def main():
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('group');p.add_argument('folder');p.add_argument('--method',choices=['appearance','chronological','hybrid'],default='appearance');p.add_argument('--scenes',type=int,default=4);p.add_argument('--gap-minutes',type=float,default=5);p.add_argument('--approx-photos-per-group',type=int,default=8);p.add_argument('--out',required=True);p.add_argument('--overwrite',action='store_true')
    p=sub.add_parser('target');p.add_argument('original');p.add_argument('--preset',choices=PRESETS,default='warm');p.add_argument('--recipe');p.add_argument('--out',required=True);p.add_argument('--overwrite',action='store_true')
    p=sub.add_parser('fit');p.add_argument('original');p.add_argument('target');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=512);p.add_argument('--iterations',type=int,default=45);p.add_argument('--solver',choices=['pattern','gauss-newton','least-squares'],default='pattern');p.add_argument('--renderer',choices=['global','selective','gamut','polynomial','spatial','grid'],default='global');p.add_argument('--regularization',type=float,default=0);p.add_argument('--overwrite',action='store_true')
    p=sub.add_parser('album');p.add_argument('groups');p.add_argument('--preset',choices=PRESETS,default='warm');p.add_argument('--recipe');p.add_argument('--target-engine',choices=['morph','pillow','pillow-selective','pillow-soft','spatial-soft','gaussian-soft'],default='morph');p.add_argument('--target-settings',help='JSON controls for the independent pillow-soft target');p.add_argument('--targets',help='Folder containing a same-name edited target for every group representative');p.add_argument('--strength',type=float,default=1.0,help='Applied recipe strength from 0 (neutral) to 1 (fully fitted)');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=512);p.add_argument('--iterations',type=int,default=45);p.add_argument('--solver',choices=['pattern','gauss-newton','least-squares'],default='pattern');p.add_argument('--renderer',choices=['global','selective','gamut','polynomial','spatial','grid'],default='global');p.add_argument('--regularization',type=float,default=0)
    p=sub.add_parser('apply');p.add_argument('groups');p.add_argument('fits',help='Album output folder containing scene fit.json files');p.add_argument('--out',required=True);p.add_argument('--strength',type=float,help='Override recorded strength, from 0 to 1');p.add_argument('--local-strength',type=float,help='Scale only grid local-light weights from 0 to 1');p.add_argument('--corrections',help='Optional JSON corrections at album, scene, and photo levels');p.add_argument('--preview-samples',type=int,default=0,help='Render only up to N full-resolution photos per scene for review');p.add_argument('--allow-changed-inputs',action='store_true',help='Explicitly allow changed photos or scene membership when replaying a fit');p.add_argument('--overwrite',action='store_true')
    p=sub.add_parser('evaluate');p.add_argument('originals');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=10);p.add_argument('--size',type=int,default=512);p.add_argument('--overwrite',action='store_true')
    sub.add_parser('prompt-help',help='List supported offline photographic prompt phrases')
    # The album parser is retained above; attach overwrite without changing defaults.
    sub.choices['album'].add_argument('--overwrite',action='store_true')
    sub.choices['album'].add_argument('--full-res-reference',action='store_true',help='Export full-size representative target and reconstruction (morph, pillow-soft, spatial-soft, gaussian-soft, or supplied targets)')
    sub.choices['album'].add_argument('--prompt',help='Comma-separated offline photographic phrases; requires pillow-soft, spatial-soft, or gaussian-soft')
    sub.choices['album'].add_argument('--target-local-light',type=float,help='Local-light amount from -0.5 to 0.5 for spatial-soft or gaussian-soft targets')
    sub.choices['album'].add_argument('--target-light-center',nargs=2,type=float,metavar=('X','Y'),help='Gaussian center as fractions of frame width and height')
    sub.choices['album'].add_argument('--target-light-radius',nargs=2,type=float,metavar=('X','Y'),help='Gaussian standard deviations as fractions of frame width and height')
    sub.choices['album'].add_argument('--preview-only',action='store_true',help='Fit representative targets and save previews without exporting every photo')
    sub.choices['album'].add_argument('--preview-samples',type=int,default=0,help='In preview mode, also render up to N full-resolution photos per group')
    sub.choices['album'].add_argument('--evaluate-transfer',action='store_true',help='Compare rendered photos with their individual deterministic or supplied targets at fitting resolution')
    a=ap.parse_args()
    if a.cmd in ('fit','album'):
        if not math.isfinite(a.regularization) or a.regularization<0:raise ValueError('regularization must be nonnegative and finite')
        if a.renderer in ('polynomial','spatial','grid'):
            if a.solver!='least-squares':raise ValueError(a.renderer+' renderer requires --solver least-squares')
            if a.renderer=='polynomial':import poly_color as model
            elif a.renderer=='spatial':import spatial_light as model
            else:import grid_light as model
            model_render=model.render;model_vec=model.deserialize;model_recipe=model.serialize
            model_fit=(lambda x,y:model.fit(x,y,regularization=a.regularization)) if a.renderer=='polynomial' else (lambda x,y:model.fit_with_metrics(x,y,regularization=a.regularization))
        elif a.renderer in ('selective','gamut'):
            if a.solver!='gauss-newton':raise ValueError(a.renderer+' renderer requires --solver gauss-newton')
            if a.renderer=='selective':
                import selective as model
                model_fit=lambda x,y:model.fit(x,y,iterations=a.iterations,regularization=a.regularization)
            else:
                if a.regularization:raise ValueError('regularization is not supported by the gamut renderer')
                import gamut_renderer as model
                model_fit=lambda x,y:model.fit(x,y,iterations=a.iterations)
            model_render=model.render;model_vec=model.vec;model_recipe=model.recipe
        else:
            if a.regularization:raise ValueError('regularization is only supported by selective, polynomial, spatial and grid renderers')
            if a.solver=='least-squares':raise ValueError('least-squares solver requires the polynomial, spatial or grid renderer')
            model_render=render;model_vec=vec;model_recipe=recipe
            model_fit=lambda x,y:run_fit(x,y,iterations=a.iterations,method=a.solver)
    if a.cmd=='prompt-help':
        import prompt_target
        print('Separate supported phrases with commas or "and"; prefix a phrase with "slightly" or "strongly" to adjust intensity.')
        for phrase in prompt_target.PHRASES:print('  '+phrase)
        print('Local lighting phrases (gaussian-soft only):')
        for phrase in prompt_target.LOCAL_PHRASES:print('  '+phrase)
    elif a.cmd=='group':
        ps=files(a.folder)
        if not ps or a.scenes<1: raise ValueError('Need at least one photo and one scene')
        protect_inputs(ps,[a.out]);protect_existing([a.out],a.overwrite)
        groups=cluster(ps,a.scenes) if a.method=='appearance' else (hybrid_groups(ps,a.gap_minutes,a.approx_photos_per_group) if a.method=='hybrid' else chronological_groups(ps,a.gap_minutes))
        dump({'version':1,'method':a.method+' baseline; manually review groups','gap_minutes':a.gap_minutes if a.method in ('chronological','hybrid') else None,'approx_photos_per_group':a.approx_photos_per_group if a.method=='hybrid' else None,'groups':groups},a.out);print(f'{len(ps)} images in {len(groups)} groups: {a.out}')
    elif a.cmd=='target':
        outputs=[a.out,str(a.out)+'.recipe.json'];protect_inputs([a.original]+([a.recipe] if a.recipe else []),outputs);protect_existing(outputs,a.overwrite)
        v=vec(json.loads(Path(a.recipe).read_text()) if a.recipe else PRESETS[a.preset]);save_render(load(a.original),v,a.out,source=a.original);dump(recipe(v),str(a.out)+'.recipe.json');print(a.out)
    elif a.cmd=='fit':
        outputs=[Path(a.out)/'reconstruction.jpg',Path(a.out)/'fit.json'];protect_inputs([a.original,a.target],outputs);protect_existing(outputs,a.overwrite)
        x=load(a.original,a.size);y=load(a.target,a.size);v,metrics=model_fit(x,y);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);save(model_render(x,v),out/'reconstruction.jpg');dump({'recipe':model_recipe(v),'renderer':a.renderer,'regularization':a.regularization,'solver':a.solver,'metrics':metrics},out/'fit.json');print(json.dumps(metrics))
    elif a.cmd=='album':
        if not math.isfinite(a.strength) or not 0<=a.strength<=1: raise ValueError('strength must be between 0 and 1')
        if a.iterations<1: raise ValueError('iterations must be positive')
        if a.preview_samples<0 or (a.preview_samples and not a.preview_only):raise ValueError('--preview-samples must be nonnegative and requires --preview-only')
        if a.targets and a.recipe: raise ValueError('Choose edited targets or a synthetic recipe, not both.')
        if a.prompt is not None and (a.target_engine not in ('pillow-soft','spatial-soft','gaussian-soft') or a.targets or a.recipe or a.target_settings):raise ValueError('--prompt requires pillow-soft, spatial-soft, or gaussian-soft and cannot be combined with --targets, --recipe or --target-settings')
        if a.target_settings and (a.target_engine not in ('pillow-soft','spatial-soft','gaussian-soft') or a.targets or a.recipe):raise ValueError('--target-settings requires pillow-soft, spatial-soft, or gaussian-soft and cannot be combined with --targets or --recipe')
        if a.target_engine in ('pillow','pillow-selective','pillow-soft','spatial-soft','gaussian-soft') and a.recipe: raise ValueError('Custom Morph recipes cannot be used with this target engine; choose target settings or a preset.')
        if a.target_local_light is not None and (a.target_engine not in ('spatial-soft','gaussian-soft') or a.targets):raise ValueError('--target-local-light requires a spatial-soft or gaussian-soft target')
        if (a.target_light_center is not None or a.target_light_radius is not None) and (a.target_engine!='gaussian-soft' or a.targets):raise ValueError('Light center and radius require a gaussian-soft target')
        light_center=a.target_light_center if a.target_light_center is not None else [.5,.5]
        light_radius=a.target_light_radius if a.target_light_radius is not None else [.18,.23]
        if not all(math.isfinite(v) and 0<=v<=1 for v in light_center):raise ValueError('Light center must be within 0..1')
        if not all(math.isfinite(v) and .03<=v<=1 for v in light_radius):raise ValueError('Light radius must be within 0.03..1')
        local_light=.2 if a.target_local_light is None else a.target_local_light
        if not math.isfinite(local_light) or not -.5<=local_light<=.5:raise ValueError('target-local-light must be between -0.5 and 0.5')
        if a.full_res_reference and not a.targets and a.target_engine in ('pillow','pillow-selective'):raise ValueError('Full-resolution reference generation is supported for morph, pillow-soft, or supplied targets')
        interpreted=None
        if a.prompt is not None:
            import prompt_target
            parsed,interpreted,local_instruction=prompt_target.parse_with_local(a.prompt);soft_custom=soft_settings(a.preset,parsed)
            if local_instruction is not None:
                if a.target_engine!='gaussian-soft':raise ValueError('Local lighting prompt phrases require gaussian-soft')
                if a.target_local_light is not None or a.target_light_center is not None:raise ValueError('Local lighting prompt conflicts with explicit local strength or center')
                local_light=local_instruction['strength'];light_center=local_instruction['center']
        else:soft_custom=soft_settings(a.preset,json.loads(Path(a.target_settings).read_text())) if a.target_settings else None
        if a.renderer in ('polynomial','spatial','grid') and a.target_engine=='morph' and not a.recipe and not a.targets:raise ValueError(a.renderer+' target generation requires --recipe or an independent target engine')
        data=json.loads(Path(a.groups).read_text()); groups=validate_groups(data)
        target=model_vec(json.loads(Path(a.recipe).read_text())) if a.recipe else (model.identity() if a.renderer in ('polynomial','spatial','grid') else model_vec(PRESETS[a.preset]));out=Path(a.out); records=[];review=[]
        samples={g['id']:preview_samples(g,a.preview_samples) for g in groups} if a.preview_only and a.preview_samples else {}
        inputs=[photo for g in groups for photo in g['photos']]+[a.groups]
        if a.recipe:inputs.append(a.recipe)
        if a.target_settings:inputs.append(a.target_settings)
        if a.targets:inputs.extend(str(Path(a.targets)/Path(g['representative']).name) for g in groups)
        outputs=[out/'report.json',out/'review.json']
        for g in groups:
            scene=out/g['id'];outputs.extend(scene/name for name in ('target.jpg','reconstruction.jpg','applied-preview.jpg','fit.json'))
            if a.full_res_reference:outputs.extend(scene/name for name in ('target-full.jpg','reconstruction-full.jpg','applied-full.jpg'))
            if not a.preview_only:outputs.extend(scene/'exports'/Path(photo).name for photo in g['photos'])
            else:outputs.extend(scene/'samples'/Path(photo).name for photo in samples.get(g['id'],[]))
        protect_inputs(inputs,outputs);protect_existing(outputs,a.overwrite)
        if a.targets:
            for g in groups:
                rep=Path(g['representative']); edit=Path(a.targets)/rep.name
                if not edit.is_file(): raise ValueError(f'Missing target for {rep}: {edit}')
                with Image.open(rep) as original, Image.open(edit) as edited:
                    if oriented_size(original)!=oriented_size(edited): raise ValueError(f'Target dimensions differ for {rep}; alignment required.')
        for g in groups:
            rep=Path(g['representative']); x=load(rep,a.size)
            if a.targets: y=load(Path(a.targets)/rep.name,a.size); engine='file'
            else:
                generators={'pillow':pillow_target,'pillow-selective':pillow_selective_target,'pillow-soft':pillow_soft_target}
                if a.target_engine=='spatial-soft':
                    import spatial_light
                    y=spatial_light.target(x,soft_settings(a.preset,soft_custom),local_light)
                elif a.target_engine=='gaussian-soft':
                    import spatial_light
                    y=spatial_light.gaussian_target(x,soft_settings(a.preset,soft_custom),local_light,light_center,light_radius)
                else:y=model_render(x,target) if a.target_engine=='morph' else (pillow_soft_target(x,a.preset,soft_custom) if a.target_engine=='pillow-soft' else generators[a.target_engine](x,a.preset))
                engine=a.target_engine
            recovered,metrics=model_fit(x,y);applied=recovered*a.strength;scene=out/g['id'];save(y,scene/'target.jpg');save(model_render(x,recovered),scene/'reconstruction.jpg');save(model_render(x,applied),scene/'applied-preview.jpg')
            manifest=[{'path':str(Path(p).resolve()),'sha256':sha256_file(p)} for p in g['photos']]
            dump({'target_engine':engine,'preset':a.preset if not a.targets and not a.recipe and not a.target_settings and a.prompt is None else None,'prompt':a.prompt,'interpreted_prompt':interpreted,'target_settings':soft_settings(a.preset,soft_custom) if engine in ('pillow-soft','spatial-soft','gaussian-soft') else None,'target_local_light':local_light if engine in ('spatial-soft','gaussian-soft') else None,'target_light_center':light_center if engine=='gaussian-soft' else None,'target_light_radius':light_radius if engine=='gaussian-soft' else None,'known_recipe':model_recipe(target) if engine=='morph' else None,'fitted_recipe':model_recipe(recovered),'applied_recipe':model_recipe(applied),'renderer':a.renderer,'regularization':a.regularization,'solver':a.solver,'strength':a.strength,'iterations':a.iterations,'metrics':metrics,'representative':str(rep),'source_manifest':manifest},scene/'fit.json')
            if a.full_res_reference:
                full=load(rep)
                if engine=='file':
                    target_path=Path(a.targets)/rep.name; target_full=load(target_path)
                    save_render(target_full,np.zeros(len(NAMES)),scene/'target-full.jpg',source=target_path)
                elif engine=='pillow-soft':
                    def soft_renderer(tile,unused):return pillow_soft_target(tile,a.preset,soft_custom)
                    save_render(full,np.zeros(len(NAMES)),scene/'target-full.jpg',renderer=soft_renderer,source=rep)
                elif engine=='spatial-soft':
                    import spatial_light
                    def spatial_target_renderer(tile,unused):return spatial_light.target(tile,soft_settings(a.preset,soft_custom),local_light)
                    spatial_target_renderer.render_tile=lambda tile,unused,start,shape:spatial_light.target_tile(tile,soft_settings(a.preset,soft_custom),local_light,start,shape)
                    save_render(full,np.zeros(len(NAMES)),scene/'target-full.jpg',renderer=spatial_target_renderer,source=rep)
                elif engine=='gaussian-soft':
                    import spatial_light
                    def gaussian_target_renderer(tile,unused):return spatial_light.gaussian_target(tile,soft_settings(a.preset,soft_custom),local_light,light_center,light_radius)
                    gaussian_target_renderer.render_tile=lambda tile,unused,start,shape:spatial_light.gaussian_target(tile,soft_settings(a.preset,soft_custom),local_light,light_center,light_radius,start,shape)
                    save_render(full,np.zeros(len(NAMES)),scene/'target-full.jpg',renderer=gaussian_target_renderer,source=rep)
                else:save_render(full,target,scene/'target-full.jpg',renderer=model_render,source=rep)
                save_render(full,recovered,scene/'reconstruction-full.jpg',renderer=model_render,source=rep)
                save_render(full,applied,scene/'applied-full.jpg',renderer=model_render,source=rep)
            selected=samples.get(g['id'],[]) if a.preview_only else g['photos']
            for path in selected:
                src=Path(path);image=load(src);diag=clipping_diagnostic(image,applied,renderer=model_render)
                if a.renderer=='polynomial':
                    import poly_color
                    diag['polynomial_out_of_gamut_fraction']=poly_color.out_of_gamut(image,applied)
                elif a.renderer=='grid':
                    import grid_light
                    diag['grid_out_of_gamut_fraction']=grid_light.out_of_gamut(image,applied)
                destination=scene/('samples' if a.preview_only else 'exports')/src.name;save_render(image,applied,destination,renderer=model_render,source=src)
                entry={'scene':g['id'],'original':str(src),'export':str(destination),'clipping':diag}
                if a.evaluate_transfer:
                    role='representative' if src.resolve()==rep.resolve() else 'other photo'
                    target_image=None;reason=None
                    if engine=='file':
                        source_target=Path(a.targets)/src.name
                        if not source_target.is_file():reason='no same-name target'
                        else:
                            with Image.open(src) as original,Image.open(source_target) as edited:
                                if oriented_size(original)!=oriented_size(edited):reason='target dimensions differ'
                            if reason is None:target_image=load(source_target,a.size)
                    sample=load(src,a.size)
                    if engine=='morph':target_image=model_render(sample,target)
                    elif engine=='pillow-soft':target_image=pillow_soft_target(sample,a.preset,soft_custom)
                    elif engine=='spatial-soft':
                        import spatial_light
                        target_image=spatial_light.target(sample,soft_settings(a.preset,soft_custom),local_light)
                    elif engine=='gaussian-soft':
                        import spatial_light
                        target_image=spatial_light.gaussian_target(sample,soft_settings(a.preset,soft_custom),local_light,light_center,light_radius)
                    elif engine in ('pillow','pillow-selective'):target_image=generators[engine](sample,a.preset)
                    if target_image is None:entry['transfer']={'role':role,'status':'unavailable','reason':reason}
                    else:
                        reconstructed=model_render(sample,applied);baseline_rmse=float(np.sqrt(np.mean((sample-target_image)**2)));transfer_rmse=float(np.sqrt(np.mean((reconstructed-target_image)**2)))
                        entry['transfer']={'role':role,'status':'evaluated','source_to_target_rmse':baseline_rmse,'output_to_target_rmse':transfer_rmse,'worse_than_source':transfer_rmse>baseline_rmse+1e-6,'size':a.size}
                review.append(entry)
            records.append({'scene':g['id'],'representative':str(rep),'photos':len(g['photos']),'sample_photos':selected if a.preview_only else None,'metrics':metrics});print(g['id'],metrics,flush=True)
        evaluated=[r['transfer'] for r in review if r.get('transfer',{}).get('status')=='evaluated' and r['transfer']['role']=='other photo']
        dump({'mode':'preview' if a.preview_only else 'album','target_engine':'file' if a.targets else a.target_engine,'prompt':a.prompt,'interpreted_prompt':interpreted,'renderer':a.renderer,'regularization':a.regularization,'strength':a.strength,'scenes':records,'export_count':0 if a.preview_only else len(review),'sample_count':len(review) if a.preview_only else 0,'clipping_flags':None if not review and a.preview_only else sum(r['clipping']['flag'] for r in review),'transfer_check':{'enabled':a.evaluate_transfer,'other_photos_evaluated':len(evaluated),'worse_than_source':sum(r['worse_than_source'] for r in evaluated)}},out/'report.json')
        note=('Preview checks cover only the selected samples, not the rest of the album.' if review else 'No album photos were checked in preview-only mode.') if a.preview_only else None
        dump({'method':'Regular spatial sampling before export encoding. High = any channel >= 254/255; low = all channels <= 1/255. Flag = either fraction increased more than 1 percentage point; heuristic, not a quality verdict.','note':note,'photos':review},out/'review.json')
    elif a.cmd=='apply':
        if a.strength is not None and (not math.isfinite(a.strength) or not 0<=a.strength<=1):raise ValueError('strength must be between 0 and 1')
        if a.local_strength is not None and (not math.isfinite(a.local_strength) or not 0<=a.local_strength<=1):raise ValueError('local-strength must be between 0 and 1')
        if a.preview_samples<0:raise ValueError('--preview-samples must be nonnegative')
        groups=validate_groups(json.loads(Path(a.groups).read_text()));out=Path(a.out);plans=[];inputs=[a.groups];outputs=[out/'report.json',out/'review.json']
        config=json.loads(Path(a.corrections).read_text()) if a.corrections else {}
        if a.corrections:inputs.append(a.corrections)
        if not isinstance(config,dict) or set(config)-{'album','scenes','photos','local_strengths'}:raise ValueError('Corrections must contain only album, scenes, photos and local_strengths objects')
        album_config=config.get('album',{});scene_config=config.get('scenes',{});photo_config=config.get('photos',{});local_config=config.get('local_strengths',{})
        if not all(isinstance(x,dict) for x in (album_config,scene_config,photo_config,local_config)):raise ValueError('Correction sections must be objects')
        group_ids={g['id'] for g in groups}
        if (set(scene_config)|set(photo_config)|set(local_config))-group_ids:raise ValueError('Corrections contain unknown scene IDs')
        for g in groups:
            fitfile=Path(a.fits)/g['id']/'fit.json';inputs.append(fitfile)
            data=json.loads(fitfile.read_text());renderer=data.get('renderer','global')
            if (a.local_strength is not None or g['id'] in local_config) and renderer!='grid':raise ValueError('Local strength requires grid fits: '+g['id'])
            local_strength=local_config.get(g['id'],a.local_strength if a.local_strength is not None else 1.)
            if isinstance(local_strength,bool) or not isinstance(local_strength,(int,float)) or not math.isfinite(local_strength) or not 0<=local_strength<=1:raise ValueError('Invalid local strength: '+g['id'])
            if Path(data['representative']).resolve()!=Path(g['representative']).resolve():raise ValueError('Representative differs from saved fit: '+g['id'])
            old_manifest=data.get('source_manifest');validation='verified'
            if old_manifest is None:
                validation='legacy-unverified';warnings.warn('Saved fit has no source checksums; cannot verify inputs: '+g['id'])
            else:
                if not isinstance(old_manifest,list):raise ValueError('Invalid saved source manifest: '+g['id'])
                saved={r['path']:r['sha256'] for r in old_manifest};current={str(Path(p).resolve()):p for p in g['photos']}
                differences=[]
                if len(saved)!=len(old_manifest) or set(saved)!=set(current):differences.append('scene membership changed')
                for path in set(saved)&set(current):
                    if sha256_file(current[path])!=saved[path]:differences.append('file changed: '+path)
                if differences and not a.allow_changed_inputs:raise ValueError('Inputs changed since fitting '+g['id']+': '+', '.join(differences)+'; re-fit or explicitly use --allow-changed-inputs')
                if differences:validation='changed-inputs-allowed';warnings.warn('Applying a fit with changed inputs: '+g['id'])
            if renderer=='global':render_fn=render;decode=vec;encode=recipe
            elif renderer=='selective':
                import selective as model
                render_fn=model.render;decode=model.vec;encode=model.recipe
            elif renderer=='gamut':
                import gamut_renderer as model
                render_fn=model.render;decode=model.vec;encode=model.recipe
            elif renderer=='polynomial':
                import poly_color as model
                render_fn=model.render;decode=model.deserialize;encode=model.serialize
            elif renderer=='spatial':
                import spatial_light as model
                render_fn=model.render;decode=model.deserialize;encode=model.serialize
            elif renderer=='grid':
                import grid_light as model
                render_fn=model.render;decode=model.deserialize;encode=model.serialize
            else:raise ValueError('Unknown saved renderer: '+str(renderer))
            if renderer in ('polynomial','spatial','grid'):
                import gamut_renderer as correction_model
                correction_fn=correction_model.render;correction_decode=correction_model.vec;correction_encode=correction_model.recipe;correction_name='gamut'
            else:correction_fn=render_fn;correction_decode=decode;correction_encode=encode;correction_name=renderer
            strength=a.strength if a.strength is not None else data.get('strength',1.)
            if not isinstance(strength,(int,float)) or not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid saved strength: '+g['id'])
            per_photo=photo_config.get(g['id'],{});scene_changes=scene_config.get(g['id'],{})
            if not isinstance(per_photo,dict) or not isinstance(scene_changes,dict):raise ValueError('Scene and photo corrections must be objects')
            valid_names={Path(p).name for p in g['photos']}
            if set(per_photo)-valid_names:raise ValueError('Unknown photo in corrections for '+g['id'])
            if not all(isinstance(v,dict) for v in per_photo.values()):raise ValueError('Each photo correction must be an object')
            album_vector=correction_decode(album_config);scene_vector=correction_decode(scene_changes);photo_vectors={name:correction_decode(v) for name,v in per_photo.items()}
            applied=decode(data['fitted_recipe'])*strength
            if renderer=='grid':applied[10:]*=local_strength
            scene=out/g['id'];inputs.extend(g['photos']);outputs.append(scene/'applied.json')
            if a.preview_samples:
                # Prefer representative and individually corrected photos, then fill with appearance-diverse samples.
                selected=[g['representative']];changed=[p for p in g['photos'] if Path(p).name in photo_vectors and np.any(photo_vectors[Path(p).name]) and p!=g['representative']]
                changed.sort(key=lambda p:-float(np.max(np.abs(photo_vectors[Path(p).name]))));selected.extend(changed[:max(0,a.preview_samples-1)])
                if len(selected)<min(a.preview_samples,len(g['photos'])):
                    for p in preview_samples(g,len(g['photos'])):
                        if p not in selected:selected.append(p)
                        if len(selected)>=min(a.preview_samples,len(g['photos'])):break
            else:selected=g['photos']
            outputs.extend(scene/('samples' if a.preview_samples else 'exports')/Path(p).name for p in selected);plans.append((g,renderer,render_fn,encode,applied,strength,local_strength,fitfile,correction_fn,correction_encode,correction_name,album_vector,scene_vector,photo_vectors,validation,selected))
        protect_inputs(inputs,outputs);protect_existing(outputs,a.overwrite);reviews=[];records=[]
        for g,renderer,render_fn,encode,applied,strength,local_strength,fitfile,correction_fn,correction_encode,correction_name,album_vector,scene_vector,photo_vectors,validation,selected in plans:
            scene=out/g['id'];dump({'renderer':renderer,'source_fit':str(fitfile),'source_validation':validation,'strength':strength,'local_strength':local_strength if renderer=='grid' else None,'applied_recipe':encode(applied),'correction_renderer':correction_name,'album_correction':correction_encode(album_vector),'scene_correction':correction_encode(scene_vector),'photo_corrections':{k:correction_encode(v) for k,v in photo_vectors.items()},'representative':g['representative'],'operation_order':['photo correction','fitted look','scene correction','album correction']},scene/'applied.json')
            for path in selected:
                src=Path(path);x=load(src);photo_vector=photo_vectors.get(src.name)
                def composed(image,unused):
                    if photo_vector is not None and np.any(photo_vector):image=correction_fn(image,photo_vector)
                    image=render_fn(image,applied)
                    if np.any(scene_vector):image=correction_fn(image,scene_vector)
                    if np.any(album_vector):image=correction_fn(image,album_vector)
                    return image
                if getattr(render_fn,'render_tile',None):
                    def composed_tile(image,unused,start,shape):
                        if photo_vector is not None and np.any(photo_vector):image=correction_fn(image,photo_vector)
                        image=render_fn.render_tile(image,applied,start,shape)
                        if np.any(scene_vector):image=correction_fn(image,scene_vector)
                        if np.any(album_vector):image=correction_fn(image,album_vector)
                        return image
                    composed.render_tile=composed_tile
                diagnostic=clipping_diagnostic(x,applied,renderer=composed)
                if renderer=='polynomial':
                    import poly_color
                    stride=max(1,max(x.shape[:2])//512);sample=x[::stride,::stride]
                    precorrected=correction_fn(sample,photo_vector) if photo_vector is not None and np.any(photo_vector) else sample
                    diagnostic['polynomial_out_of_gamut_fraction']=poly_color.out_of_gamut(precorrected,applied)
                elif renderer=='grid':
                    import grid_light
                    pre=(lambda sample:correction_fn(sample,photo_vector)) if photo_vector is not None and np.any(photo_vector) else None
                    diagnostic['grid_out_of_gamut_fraction']=grid_light.out_of_gamut(x,applied,pre_transform=pre)
                dest=scene/('samples' if a.preview_samples else 'exports')/src.name;save_render(x,applied,dest,renderer=composed,source=src);reviews.append({'scene':g['id'],'original':str(src),'export':str(dest),'clipping':diagnostic})
            records.append({'scene':g['id'],'photos':len(g['photos']),'sample_photos':selected if a.preview_samples else None,'renderer':renderer,'source_validation':validation,'strength':strength});print(g['id'],len(selected),'photos',flush=True)
        dump({'mode':'preview' if a.preview_samples else 'album','source_fits':str(Path(a.fits)),'scenes':records,'export_count':0 if a.preview_samples else len(reviews),'sample_count':len(reviews) if a.preview_samples else 0,'clipping_flags':sum(r['clipping']['flag'] for r in reviews)},out/'report.json')
        dump({'method':'Regular spatial sampling before export encoding. High = any channel >= 254/255; low = all channels <= 1/255. Flag = either fraction increased more than 1 percentage point; heuristic, not a quality verdict.','note':'Preview checks cover only the selected samples, not the rest of the album.' if a.preview_samples else None,'photos':reviews},out/'review.json')
    elif a.cmd=='evaluate':
        if a.limit<1:raise ValueError('limit must be positive')
        out=Path(a.out);sources=files(a.originals);inputs=sources+[Path(a.edits)/s.name for s in sources if (Path(a.edits)/s.name).exists()]
        outputs=[out/'report.json']+[out/'reconstructions'/s.name for s in sources]+[out/f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(min(a.limit,len(sources))/12))]
        protect_inputs(inputs,outputs);protect_existing(outputs,a.overwrite)
        out.mkdir(parents=True,exist_ok=True);rows=[];visuals=[]
        for src in sources:
            dst=Path(a.edits)/src.name
            if not dst.exists(): continue
            with Image.open(src) as x, Image.open(dst) as y:
                if oriented_size(x) != oriented_size(y):
                    rows.append({'file':src.name,'status':'skipped: dimensions differ; alignment required'});continue
            x=load(src,a.size);y=load(dst,a.size);v,metrics=fit(x,y);reconstruction=render(x,v);save(reconstruction,out/'reconstructions'/src.name);visuals.append((src.name,(x,y,reconstruction)));rows.append({'file':src.name,'status':'fit; equal dimensions do not guarantee registration','recipe':recipe(v),'metrics':metrics})
            print(src.name,metrics['rmse'],flush=True)
            if sum(r['status'].startswith('fit;') for r in rows)>=a.limit:break
        pages=comparison_sheet(visuals,out);dump({'results':rows,'comparison_sheets':pages},out/'report.json')
if __name__=='__main__': main()
