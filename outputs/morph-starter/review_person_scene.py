#!/usr/bin/env python3
"""Review full-resolution person-scene exports with side-by-side thumbnails."""
import argparse,html,json,math,os
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image,ImageDraw
import morph


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('report');p.add_argument('--out',required=True);p.add_argument('--per-page',type=int,default=5);a=p.parse_args()
 if not 1<=a.per_page<=10:raise ValueError('per-page must be 1–10')
 report=json.loads(Path(a.report).read_text());rows=report['photos']
 if not rows:raise ValueError('Report contains no photos')
 inputs=[a.report];seen=set();entries=[]
 for row in rows:
  source=str(Path(row['source']).resolve());output=str(Path(row['output']).resolve())
  if source in seen or source==output:raise ValueError('Duplicate source or output equals source')
  seen.add(source)
  with Image.open(source) as original,Image.open(output) as edited:
   if morph.oriented_size(original)!=morph.oriented_size(edited):raise ValueError('Export dimensions differ from source: '+source)
   has_icc=bool(edited.info.get('icc_profile'))
  inputs.extend([source,output]);entries.append({'source':source,'output':output,'review_flag':bool(row.get('review_flag')),'icc_profile_present':has_icc})
 out=Path(a.out);pages=math.ceil(len(entries)/a.per_page);outputs=[out/'index.html',out/'index.json']+[out/f'comparison-{i+1:02d}.jpg' for i in range(pages)];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False)
 if out.exists():raise ValueError('Output directory already exists')
 out.mkdir(parents=True);parts=['<!doctype html><meta charset="utf-8"><title>Scene edit review</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;background:#f5f5f3;color:#222}article{background:white;padding:16px;margin:20px 0;border-radius:8px}img{max-width:100%;height:auto}</style><h1>Scene edit review</h1><p>Comparison sheets are thumbnails. Links below open the exported full-resolution images. Inspect the person masks separately.</p>']
 for start in range(0,len(entries),a.per_page):
  subset=entries[start:start+a.per_page];sheet=Image.new('RGB',(800,260*len(subset)),'white');draw=ImageDraw.Draw(sheet)
  for i,row in enumerate(subset):
   label=Path(row['source']).name+(' [mask nearly empty: inspect]' if row['review_flag'] else '')
   draw.text((8,i*260+3),label+'  |  original / edited',fill='black')
   for j,path in enumerate((row['source'],row['output'])):
    x=morph.load(path,360);im=Image.fromarray(np.uint8(x*255+.5));im.thumbnail((380,230));sheet.paste(im,(j*400+10+(380-im.width)//2,i*260+24))
  name=f'comparison-{start//a.per_page+1:02d}.jpg';sheet.save(out/name,quality=92);parts.append(f'<article><h2>Photos {start+1}–{start+len(subset)}</h2><img src="{html.escape(name,quote=True)}" alt="Original and edited scene photos"><ul>')
  for row in subset:
   href=html.escape(quote(os.path.relpath(row['output'],out.resolve()),safe='/'),quote=True);label=html.escape(Path(row['output']).name);parts.append(f'<li><a href="{href}">{label} — full-resolution export</a></li>')
  parts.append('</ul></article>')
 parts.append('</html>');(out/'index.html').write_text(''.join(parts),encoding='utf-8');morph.dump({'scope':'Thumbnail comparisons and links to full-resolution scene exports; dimensions checked, ICC presence reported. Does not verify perceptual quality, correct masks, or clipping.','scene':report['scene'],'photos':entries},out/'index.json');print(len(entries),'photos reviewed; missing ICC:',sum(not e['icc_profile_present'] for e in entries))
if __name__=='__main__':main()
