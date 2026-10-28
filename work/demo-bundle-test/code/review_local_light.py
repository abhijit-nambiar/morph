#!/usr/bin/env python3
"""Review grid local-light strengths by scene and compile manual selections."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import grid_light,morph

FIELDS=['scene','local_strength']
LEVELS=(0.,.5,1.)


def build(a):
    if not 1<=a.samples<=8 or not 64<=a.size<=600:raise ValueError('samples must be 1..8 and size 64..600')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));panels=[];inputs=[a.groups];decisions=[]
    for group in groups:
        fitfile=Path(a.fits)/group['id']/'fit.json';inputs.append(fitfile);fit=json.loads(fitfile.read_text())
        if fit.get('renderer')!='grid':raise ValueError('Local-light review requires grid fits: '+group['id'])
        if Path(fit['representative']).resolve()!=Path(group['representative']).resolve():raise ValueError('Representative differs from saved fit: '+group['id'])
        strength=fit.get('strength',1.)
        if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid saved strength')
        weights=grid_light.deserialize(fit['fitted_recipe'])*strength
        selected=morph.preview_samples(group,min(a.samples,len(group['photos'])))
        decisions.append({'scene':group['id'],'local_strength':''})
        for path in selected:
            inputs.append(path);x=morph.load(path,a.size);images=[x]
            for level in LEVELS:
                w=weights.copy();w[10:]*=level;images.append(grid_light.render(x,w))
            panels.append((group['id'],Path(path).name,images))
    out=Path(a.out);outputs=[out/'decisions.csv',out/'index.json']+[out/f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(len(panels)/6))]
    morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,a.overwrite);out.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(panels),6):
        subset=panels[start:start+6];sheet=Image.new('RGB',(1000,205*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,(scene,name,images) in enumerate(subset):
            draw.text((5,i*205+3),f'{scene} / {name}: original | local 0 | local 0.5 | local 1',fill='black')
            for j,image in enumerate(images):
                im=Image.fromarray(np.uint8(np.clip(image*255+.5,0,255)));im.thumbnail((238,175));sheet.paste(im,(j*250+6+(238-im.width)//2,i*205+24))
        filename=f'comparison-{start//6+1:02d}.jpg';sheet.save(out/filename,quality=90);pages.append(filename)
    with open(out/'decisions.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(decisions)
    morph.dump({'scope':'Thumbnail review only; sampled photos, not the entire album. No other correction layers are applied. Full-resolution sample export via apply is needed to inspect detail.','source_validation':'Source checksum and membership validation occurs in apply, not in this thumbnail review.','scenes':len(groups),'photos':len(panels),'comparison_sheets':pages,'decisions':'decisions.csv','choices':'Any value from 0 to 1; blank scenes are omitted on compile.'},out/'index.json')
    print(len(groups),'scenes,',len(panels),'sample photos; decisions initially blank')


def compile_(a):
    with open(a.decisions,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS:raise ValueError('Unexpected decision CSV columns')
        rows=list(reader)
    seen=set();values={};blank=0
    for row in rows:
        scene=row['scene'].strip()
        if not scene or scene in seen:raise ValueError('Missing or duplicate scene')
        seen.add(scene);value=row['local_strength'].strip()
        if not value:blank+=1;continue
        try:value=float(value)
        except ValueError:raise ValueError('Local strength must be numeric')
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Local strength must be between 0 and 1')
        values[scene]=value
    outputs=[a.out,str(a.out)+'.summary.json'];morph.protect_inputs([a.decisions],outputs);morph.protect_existing(outputs,a.overwrite)
    morph.dump({'local_strengths':values},a.out);morph.dump({'selected':len(values),'unreviewed':blank},str(a.out)+'.summary.json');print(len(values),'scene selections;',blank,'unreviewed')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('groups');b.add_argument('fits');b.add_argument('--out',required=True);b.add_argument('--samples',type=int,default=3);b.add_argument('--size',type=int,default=300);b.add_argument('--overwrite',action='store_true')
    c=sub.add_parser('compile');c.add_argument('decisions');c.add_argument('--out',required=True);c.add_argument('--overwrite',action='store_true')
    args=p.parse_args();build(args) if args.command=='build' else compile_(args)
if __name__=='__main__':main()
