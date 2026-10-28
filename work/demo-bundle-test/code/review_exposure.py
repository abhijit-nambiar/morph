#!/usr/bin/env python3
"""Review suggested exposure changes and compile only explicitly accepted ones."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,gamut_renderer,poly_color,spatial_light,grid_light

FIELDS=['scene','photo','source','proposed_exposure','decision']

def build(a):
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));gmap={g['id']:g for g in groups};audit=json.loads(Path(a.audit).read_text());candidates=[r for r in audit['photos'] if r['eligible']];out=Path(a.out);fits={};records=[];panels=[]
    for row in candidates:
        scene=row['scene'];g=gmap.get(scene)
        if g is None or row['photo'] not in g['photos']:raise ValueError('Suggestion no longer belongs to a scene: '+row['photo'])
        if scene not in fits:
            fit=json.loads((Path(a.fits)/scene/'fit.json').read_text());renderer=fit.get('renderer','global')
            if Path(fit['representative']).resolve()!=Path(g['representative']).resolve():raise ValueError('Representative differs from saved fit: '+scene)
            if renderer=='gamut':fn=gamut_renderer.render;param=gamut_renderer.vec(fit['fitted_recipe'])
            elif renderer=='polynomial':fn=poly_color.render;param=poly_color.deserialize(fit['fitted_recipe'])
            elif renderer=='spatial':fn=spatial_light.render;param=spatial_light.deserialize(fit['fitted_recipe'])
            elif renderer=='grid':fn=grid_light.render;param=grid_light.deserialize(fit['fitted_recipe'])
            else:raise ValueError('Only gamut, polynomial, spatial, and grid fits are supported: '+scene)
            strength=fit.get('strength',1.)
            if not isinstance(strength,(int,float)) or not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid saved strength')
            fits[scene]=(fn,param*strength,fit.get('source_manifest'))
        fn,param,manifest=fits[scene];photo=row['photo'];x=morph.load(photo,a.size)
        correction=float(row['proposed_exposure'])
        if not math.isfinite(correction) or abs(correction)>.5:raise ValueError('Invalid proposed exposure')
        scene_only=fn(x,param);adjusted=fn(gamut_renderer.render(x,gamut_renderer.vec({'exposure':correction})),param)
        records.append({'scene':scene,'photo':Path(photo).name,'source':photo,'proposed_exposure':correction,'decision':''});panels.append((scene,Path(photo).name,correction,(x,scene_only,adjusted)))
    if not records:raise ValueError('No suggestions to review')
    outputs=[out/'decisions.csv',out/'index.json']+[out/f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(len(records)/10))]
    inputs=[a.groups,a.audit]+[r['source'] for r in records]+[Path(a.fits)/s/'fit.json' for s in fits];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,a.overwrite);out.mkdir(parents=True,exist_ok=True)
    pages=[]
    for first in range(0,len(panels),10):
        subset=panels[first:first+10];sheet=Image.new('RGB',(750,200*len(subset)),'white');d=ImageDraw.Draw(sheet)
        for i,(scene,name,value,imgs) in enumerate(subset):
            d.text((5,200*i+3),f'{scene} / {name}: original | scene look | suggested {value:+.3f}',fill='black')
            for j,arr in enumerate(imgs):
                im=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));im.thumbnail((235,170));sheet.paste(im,(j*250+7+(235-im.width)//2,200*i+24))
        name=f'comparison-{first//10+1:02d}.jpg';sheet.save(out/name,quality=90);pages.append(name)
    with open(out/'decisions.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
    morph.dump({'scope':'Preview thumbnails only; use full-resolution sample export to inspect detail. Blank decisions are not accepted.','fit_validation':'Source checksums and membership are verified by the apply command, not this thumbnail review.','candidates':len(records),'comparison_sheets':pages,'decision_file':'decisions.csv'},out/'index.json')
    print(len(records),'candidates,',len(pages),'thumbnail sheets; decisions initially blank')

def compile_(a):
    with open(a.decisions,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS:raise ValueError('Unexpected decision CSV columns')
        rows=list(reader)
    result={'photos':{}};seen=set();counts={'accepted':0,'rejected':0,'unreviewed':0}
    for r in rows:
        scene=r['scene'];photo=r['photo'];key=(scene,photo)
        if key in seen:raise ValueError('Duplicate decision for '+scene+'/'+photo)
        seen.add(key)
        if not scene or not photo or Path(photo).name!=photo:raise ValueError('Invalid scene or photo name')
        decision=r['decision'].strip().lower()
        if decision not in ('','accept','reject'):raise ValueError('Decision must be blank, accept, or reject')
        try:value=float(r['proposed_exposure'])
        except ValueError:raise ValueError('Invalid proposed exposure')
        if not math.isfinite(value) or abs(value)>.5:raise ValueError('Proposed exposure must be finite and within -0.5 to 0.5')
        if decision=='accept':result['photos'].setdefault(scene,{})[photo]={'exposure':value};counts['accepted']+=1
        elif decision=='reject':counts['rejected']+=1
        else:counts['unreviewed']+=1
    outputs=[a.out,str(a.out)+'.summary.json'];morph.protect_inputs([a.decisions],outputs);morph.protect_existing(outputs,a.overwrite);morph.dump(result,a.out);morph.dump(counts,str(a.out)+'.summary.json');print(json.dumps(counts))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('groups');b.add_argument('fits');b.add_argument('audit');b.add_argument('--out',required=True);b.add_argument('--size',type=int,default=300);b.add_argument('--overwrite',action='store_true')
    c=sub.add_parser('compile');c.add_argument('decisions');c.add_argument('--out',required=True);c.add_argument('--overwrite',action='store_true')
    a=p.parse_args();build(a) if a.command=='build' else compile_(a)
if __name__=='__main__':main()
