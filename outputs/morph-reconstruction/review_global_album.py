#!/usr/bin/env python3
"""Create a browsable review index for global/grid album replay."""
import argparse,html,json,math,os
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image,ImageDraw
import morph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('review');p.add_argument('--out',required=True);p.add_argument('--per-page',type=int,default=10);a=p.parse_args()
    if not 1<=a.per_page<=30:raise ValueError('per-page must be 1..30')
    rows=json.loads(Path(a.review).read_text())['photos']
    if not rows:raise ValueError('No photos')
    out=Path(a.out);pages=[f'comparison-{i+1:02d}.jpg' for i in range(math.ceil(len(rows)/a.per_page))];outputs=[out/'index.html',out/'index.json']+[out/n for n in pages];inputs=[a.review]+[p for row in rows for p in (row['original'],row['export'])];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False);out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),a.per_page):
        subset=rows[start:start+a.per_page];sheet=Image.new('RGB',(620,175*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(subset):
            y=i*175;flag=' | near-limit flag' if r['clipping']['flag'] else ''
            draw.text((5,y+2),f"{r['scene']} {Path(r['original']).name}: original / edited"+flag,fill=(150,30,30) if flag else (20,20,20))
            for j,path in enumerate((r['original'],r['export'])):
                x=morph.load(path,300);im=Image.fromarray(np.uint8(x*255+.5));im.thumbnail((295,150));sheet.paste(im,(j*310+5+(295-im.width)//2,y+20))
        sheet.save(out/pages[start//a.per_page],quality=90)
    links=[]
    for row in rows:
        relative=os.path.relpath(Path(row['export']).resolve(),out.resolve());href=html.escape(quote(relative,safe='/'));label=html.escape(row['scene']+' / '+Path(row['export']).name)
        links.append(f'<li><a href="{href}">{label}</a></li>')
    page_links=''.join(f'<li><a href="{html.escape(quote(page))}">{html.escape(page)}</a></li>' for page in pages)
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Album review</title><body><h1>Album review</h1><p>Comparison sheets are thumbnails. Photo links open full-resolution exports.</p><h2>Comparison sheets</h2><ul>'+page_links+'</ul><h2>Full-resolution photos</h2><ul>'+''.join(links)+'</ul></body></html>',encoding='utf-8')
    morph.dump({'scope':'Thumbnail overview; photo links point to full-resolution outputs. Clipping flags are heuristic.','photos':rows,'sheets':pages},out/'index.json');print(len(rows),'photos in',len(pages),'thumbnail sheets')
if __name__=='__main__':main()
