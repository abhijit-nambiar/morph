#!/usr/bin/env python3
"""Create thumbnail sheets for inspecting checksum-verified face boxes."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,face_scene_pipeline


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('detections');p.add_argument('--out',required=True);p.add_argument('--per-page',type=int,default=12);a=p.parse_args()
    if not 1<=a.per_page<=30:raise ValueError('per-page must be 1..30')
    rows=json.loads(Path(a.detections).read_text());prepared=[];seen=set()
    if not isinstance(rows,list) or not rows:raise ValueError('No detections')
    for row in rows:
        path=str(Path(row['path']).resolve())
        if path in seen:raise ValueError('Duplicate detection: '+path)
        seen.add(path)
        if row.get('source_sha256')!=morph.sha256_file(path):raise ValueError('Missing checksum or photo changed: '+path)
        boxes=face_scene_pipeline.checked_faces(row.get('faces'),allow_empty=True)
        prepared.append((path,boxes,row.get('error')))
    out=Path(a.out);outputs=[out/'index.json']+[out/f'detections-{i+1:02d}.jpg' for i in range(math.ceil(len(rows)/a.per_page))]
    morph.protect_inputs([a.detections]+[p for p,_,_ in prepared],outputs);morph.protect_existing(outputs,False);out.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(prepared),a.per_page):
        subset=prepared[start:start+a.per_page];columns=3;cellw=300;cellh=225;sheet=Image.new('RGB',(columns*cellw,math.ceil(len(subset)/columns)*cellh),'white');draw=ImageDraw.Draw(sheet)
        for i,(path,boxes,error) in enumerate(subset):
            x=(i%columns)*cellw;y=(i//columns)*cellh;arr=morph.load(path,280);im=Image.fromarray(np.uint8(arr*255+.5));im.thumbnail((280,190));left=x+(cellw-im.width)//2;top=y+27;sheet.paste(im,(left,top))
            label=f'{Path(path).name} | {len(boxes)} boxes'+(' | detection error' if error else (' | review: no detections' if not boxes else ''))
            draw.text((x+4,y+4),label,fill='black')
            for b in boxes:draw.rectangle((left+b['x']*im.width,top+b['y']*im.height,left+(b['x']+b['width'])*im.width,top+(b['y']+b['height'])*im.height),outline='red',width=2)
        filename=f'detections-{start//a.per_page+1:02d}.jpg';sheet.save(out/filename,quality=90);pages.append(filename)
    morph.dump({'scope':'Thumbnail face-box review, not an accuracy verdict. Red rectangles indicate detections; no boxes may indicate a missed face.','photos':len(rows),'sheets':pages},out/'index.json');print(len(rows),'photos,',len(pages),'thumbnail sheets')
if __name__=='__main__':main()
