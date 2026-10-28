#!/usr/bin/env python3
"""Create review sheets of estimated original crops and edited targets."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,crop_match

def main():
    p=argparse.ArgumentParser();p.add_argument('originals');p.add_argument('edits');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];visual=[]
    for src in morph.files(a.originals):
        dst=Path(a.edits)/src.name
        if not dst.exists():continue
        with Image.open(src) as x,Image.open(dst) as y:
            if morph.oriented_size(x)==morph.oriented_size(y):continue
        x=morph.load(src,1200);y=morph.load(dst,1200);m=crop_match.match(x,y);b=m['box'];h,w=x.shape[:2];crop=x[int(b[1]*h):int(b[3]*h),int(b[0]*w):int(b[2]*w)];rows.append({'file':src.name,'original':str(src),'target':str(dst),'estimate':m});visual.append((src.name,crop,y,m['score']));print(src.name,round(m['score'],3),flush=True)
    pages=[]
    for start in range(0,len(visual),10):
        subset=visual[start:start+10];sheet=Image.new('RGB',(600,len(subset)*205),'white');d=ImageDraw.Draw(sheet)
        for i,(name,crop,target,score) in enumerate(subset):
            y=i*205;d.text((8,y+3),f'{name}: original estimated crop / edited target (score {score:.3f})',fill='black')
            for j,arr in enumerate((crop,target)):
                im=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));im.thumbnail((288,175));sheet.paste(im,(j*300+6+(288-im.width)//2,y+24))
        path=f'comparison-{start//10+1:02d}.jpg';sheet.save(out/path,quality=90);pages.append(path)
    morph.dump({'warning':'Provisional automated crop estimates, not validated matches. Scores are uncalibrated; inspect sheets. Rotations and local warps are unsupported. Do not use crops as final exports.','rows':rows,'comparison_sheets':pages},out/'report.json')
if __name__=='__main__':main()
