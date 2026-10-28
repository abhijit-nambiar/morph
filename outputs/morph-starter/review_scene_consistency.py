#!/usr/bin/env python3
"""Review flagged face-album photos with manual exposure options."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,gamut_renderer,face_core,face_scene_pipeline

FIELDS=['scene','photo','source','exposure']


def build(a):
    if not 64<=a.size<=600 or not math.isfinite(a.step) or not 0<a.step<=.5:raise ValueError('Invalid review size or step')
    audit=json.loads(Path(a.audit).read_text());candidates=[r for r in audit['photos'] if r.get('review_flag') is True];fits={};panels=[];decisions=[];inputs=[a.audit];seen=set()
    for row in candidates:
        scene=row['scene'];source=str(Path(row['source']).resolve());key=(scene,source)
        if key in seen:raise ValueError('Duplicate candidate')
        seen.add(key)
        if not isinstance(scene,str) or not __import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',scene):raise ValueError('Invalid scene')
        if scene not in fits:
            fitfile=Path(a.fits)/scene/'fit.json';fit,w,sources=face_scene_pipeline.validate_fit(fitfile)
            if fit['scene']!=scene:raise ValueError('Scene mismatch')
            reportfile=Path(a.album)/scene/'report.json';report=json.loads(reportfile.read_text())
            if report['scene']!=scene or Path(report['source_fit']).resolve()!=fitfile.resolve():raise ValueError('Album report references a different fit')
            corrections=report.get('corrections',{})
            if any(corrections.get(k,{}) for k in ('album','scene','photos')):raise ValueError('Review currently requires an album rendered without corrections')
            fits[scene]=(w,dict(sources),{str(Path(r['source']).resolve()):r for r in report['photos']});inputs.extend([fitfile,reportfile]);inputs.extend(p for p,_ in sources)
        w,sources,rendered=fits[scene]
        if source not in sources or source not in rendered:raise ValueError('Candidate not in fitted and rendered scene')
        exported=str(Path(rendered[source]['output']).resolve())
        if exported!=str(Path(row['export']).resolve()):raise ValueError('Audit export differs from scene report')
        x=morph.load(source,a.size);boxes=sources[source]
        images=[morph.load(exported,a.size)]
        for value in (-a.step,a.step):
            adjusted=gamut_renderer.render(x,gamut_renderer.vec({'exposure':value}));images.append(face_core.render(adjusted,boxes,w))
        panels.append((scene,Path(source).name,images));decisions.append({'scene':scene,'photo':Path(source).name,'source':source,'exposure':''});inputs.append(exported)
    out=Path(a.out);outputs=[out/'index.json',out/'decisions.csv']+[out/f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(len(panels)/10))];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False);out.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(panels),10):
        subset=panels[start:start+10];sheet=Image.new('RGB',(750,200*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,(scene,name,images) in enumerate(subset):
            draw.text((5,200*i+3),f'{scene} / {name}: current | exposure {-a.step:+.2f} | exposure {a.step:+.2f}',fill='black')
            for j,arr in enumerate(images):
                image=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));image.thumbnail((235,170));sheet.paste(image,(j*250+7+(235-image.width)//2,200*i+24))
        filename=f'comparison-{start//10+1:02d}.jpg';sheet.save(out/filename,quality=90);pages.append(filename)
    with open(out/'decisions.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(decisions)
    morph.dump({'scope':'Thumbnail comparison only. Variants are examples, not recommendations; choose values manually and inspect full-resolution previews. This tool only supports runs without existing corrections.','step':a.step,'candidates':len(decisions),'sheets':pages,'decisions':'decisions.csv'},out/'index.json');print(len(decisions),'photos; decisions initially blank')


def compile_(a):
    with open(a.decisions,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS:raise ValueError('Unexpected decisions columns')
        rows=list(reader)
    result={'photos':{}};seen=set();blank=0
    for row in rows:
        scene=row['scene'].strip();photo=row['photo'].strip();key=(scene,photo)
        if not scene or not photo or Path(photo).name!=photo or key in seen:raise ValueError('Invalid or duplicate photo')
        seen.add(key);value=row['exposure'].strip()
        if not value:blank+=1;continue
        try:amount=float(value)
        except ValueError:raise ValueError('Exposure must be numeric')
        if not math.isfinite(amount) or abs(amount)>.5:raise ValueError('Exposure must be within -0.5..0.5')
        if amount!=0:result['photos'].setdefault(scene,{})[photo]={'exposure':amount}
    morph.protect_inputs([a.decisions],[a.out]);morph.protect_existing([a.out],False);morph.dump(result,a.out);print(sum(len(v) for v in result['photos'].values()),'selected corrections;',blank,'blank')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('fits');b.add_argument('album');b.add_argument('audit');b.add_argument('--out',required=True);b.add_argument('--size',type=int,default=300);b.add_argument('--step',type=float,default=.2)
    c=sub.add_parser('compile');c.add_argument('decisions');c.add_argument('--out',required=True)
    a=p.parse_args();build(a) if a.command=='build' else compile_(a)
if __name__=='__main__':main()
