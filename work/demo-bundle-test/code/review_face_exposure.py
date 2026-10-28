#!/usr/bin/env python3
"""Build thumbnail comparisons for review-only face-scene exposure suggestions."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,gamut_renderer,face_core,face_scene_pipeline

FIELDS=['scene','photo','source','proposed_exposure','decision']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits');p.add_argument('audit');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=300);a=p.parse_args()
    if not 64<=a.size<=600:raise ValueError('size must be 64..600')
    audit=json.loads(Path(a.audit).read_text());candidates=[r for r in audit['photos'] if r.get('eligible')];fits={};panels=[];records=[];inputs=[a.audit];seen=set()
    for row in candidates:
        scene=row['scene'];photo=str(Path(row['photo']).resolve());key=(scene,photo)
        if key in seen:raise ValueError('Duplicate candidate')
        seen.add(key)
        if not isinstance(scene,str) or not __import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',scene):raise ValueError('Invalid scene')
        if scene not in fits:
            fitfile=Path(a.fits)/scene/'fit.json';fit,w,sources=face_scene_pipeline.validate_fit(fitfile)
            if fit['scene']!=scene:raise ValueError('Scene mismatch')
            fits[scene]=(w,dict(sources));inputs.append(fitfile);inputs.extend(p for p,_ in sources)
        w,sources=fits[scene]
        if photo not in sources:raise ValueError('Photo not in fitted scene')
        try:amount=float(row['proposed_exposure'])
        except (KeyError,TypeError,ValueError):raise ValueError('Invalid suggested exposure')
        if not math.isfinite(amount) or abs(amount)>.5:raise ValueError('Suggested exposure out of bounds')
        x=morph.load(photo,a.size);boxes=sources[photo];base=face_core.render(x,boxes,w);adjusted=face_core.render(gamut_renderer.render(x,gamut_renderer.vec({'exposure':amount})),boxes,w)
        records.append({'scene':scene,'photo':Path(photo).name,'source':photo,'proposed_exposure':amount,'decision':''});panels.append((scene,Path(photo).name,amount,(x,base,adjusted)))
    if not panels:raise ValueError('No eligible suggestions')
    out=Path(a.out);outputs=[out/'index.json',out/'decisions.csv']+[out/f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(len(panels)/10))];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False);out.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(panels),10):
        subset=panels[start:start+10];sheet=Image.new('RGB',(750,200*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,(scene,name,amount,images) in enumerate(subset):
            draw.text((5,i*200+3),f'{scene} / {name}: original | scene look | suggestion {amount:+.3f}',fill='black')
            for j,array in enumerate(images):
                image=Image.fromarray(np.uint8(np.clip(array*255+.5,0,255)));image.thumbnail((235,170));sheet.paste(image,(j*250+7+(235-image.width)//2,i*200+24))
        filename=f'comparison-{start//10+1:02d}.jpg';sheet.save(out/filename,quality=90);pages.append(filename)
    with open(out/'decisions.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
    morph.dump({'scope':'Thumbnail review; no album or scene correction layers are shown. Inspect full-resolution samples before export. Blank decisions are not accepted.','candidates':len(records),'comparison_sheets':pages,'decisions':'decisions.csv'},out/'index.json');print(len(records),'candidates; decisions initially blank')
if __name__=='__main__':main()
