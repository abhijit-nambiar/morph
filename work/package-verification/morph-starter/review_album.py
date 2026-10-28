#!/usr/bin/env python3
"""Generate source/export contact sheets from an album review report."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph

def main():
    p=argparse.ArgumentParser();p.add_argument('review');p.add_argument('--out',required=True);p.add_argument('--per-page',type=int,default=12);a=p.parse_args()
    if not 1<=a.per_page<=30:p.error('--per-page must be 1..30')
    data=json.loads(Path(a.review).read_text());rows=data['photos'];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(rows),a.per_page):
        subset=rows[start:start+a.per_page];sheet=Image.new('RGB',(620,175*len(subset)),'white');d=ImageDraw.Draw(sheet)
        for i,r in enumerate(subset):
            y=i*175;mark=' [near-limit flag]' if r['clipping']['flag'] else ''
            d.text((5,y+2),f"{r['scene']} {Path(r['original']).name}: original / exported"+mark,fill=(150,30,30) if mark else (20,20,20))
            for j,path in enumerate((r['original'],r['export'])):
                x=morph.load(path,300);im=Image.fromarray(np.uint8(x*255+.5));im.thumbnail((295,150));sheet.paste(im,(j*310+5+(295-im.width)//2,y+20))
        name=f'comparison-{start//a.per_page+1:02d}.jpg';sheet.save(out/name,quality=90);pages.append(name)
    morph.dump({'scope':'Side-by-side thumbnail review; near-limit flags are heuristics, not verdicts.','photos':len(rows),'comparison_sheets':pages},out/'index.json');print('created',len(pages),'sheets covering',len(rows),'photos')
if __name__=='__main__':main()
