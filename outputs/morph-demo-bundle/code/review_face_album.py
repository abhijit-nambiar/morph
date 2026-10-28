#!/usr/bin/env python3
"""Create thumbnail contact sheets linking to full-resolution face-album exports."""
import argparse,json,math,re,html,os
from urllib.parse import quote
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('album');p.add_argument('--out',required=True);p.add_argument('--per-page',type=int,default=10);a=p.parse_args()
    if not 1<=a.per_page<=30:raise ValueError('per-page must be 1..30')
    album=Path(a.album);report=json.loads((album/'report.json').read_text());scenes=report['scenes'];rows=[];inputs=[album/'report.json']
    for scene in scenes:
        if not isinstance(scene,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',scene):raise ValueError('Invalid scene ID')
        path=album/scene/'report.json';data=json.loads(path.read_text());inputs.append(path)
        if data['scene']!=scene:raise ValueError('Scene mismatch')
        for r in data['photos']:
            row={'scene':scene,'source':r['source'],'export':r['output'],'faces':r['faces']};inputs.extend((row['source'],row['export']));rows.append(row)
    if not rows:raise ValueError('No photos to review')
    out=Path(a.out);outputs=[out/'index.json',out/'index.html']+[out/f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(len(rows)/a.per_page))];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False);out.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(rows),a.per_page):
        subset=rows[start:start+a.per_page];sheet=Image.new('RGB',(620,175*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(subset):
            y=i*175;flag=' | no faces detected: review' if not row['faces'] else ''
            draw.text((5,y+2),f"{row['scene']} {Path(row['source']).name}: original / edited"+flag,fill=(150,30,30) if flag else (20,20,20))
            for j,path in enumerate((row['source'],row['export'])):
                x=morph.load(path,300);im=Image.fromarray(np.uint8(x*255+.5));im.thumbnail((295,150));sheet.paste(im,(j*310+5+(295-im.width)//2,y+20))
        name=f'comparison-{start//a.per_page+1:02d}.jpg';sheet.save(out/name,quality=90);pages.append(name)
    morph.dump({'scope':'Thumbnail overview only. Use the paths listed for full-resolution files. No-face detection warning is not a claim that a photo contains no faces.','mode':report['mode'],'photos':rows,'comparison_sheets':pages},out/'index.json')
    page_links=''.join(f'<li><a href="{html.escape(quote(name))}">{html.escape(name)}</a></li>' for name in pages)
    photo_links=[]
    for row in rows:
        relative=os.path.relpath(Path(row['export']).resolve(),out.resolve());href=html.escape(quote(relative,safe='/'))
        name=html.escape(row['scene']+' / '+Path(row['export']).name);flag=' — no faces detected; inspect' if not row['faces'] else ''
        photo_links.append(f'<li><a href="{href}">{name}</a>{flag}</li>')
    markup='<!doctype html><html lang="en"><meta charset="utf-8"><title>Album review</title><body><h1>Album review</h1><p>Comparison sheets are thumbnails. The links below open individual full-resolution exports.</p><h2>Comparison sheets</h2><ul>'+page_links+'</ul><h2>Full-resolution photos</h2><ul>'+''.join(photo_links)+'</ul></body></html>'
    (out/'index.html').write_text(markup,encoding='utf-8');print(len(rows),'photos in',len(pages),'thumbnail sheets')
if __name__=='__main__':main()
