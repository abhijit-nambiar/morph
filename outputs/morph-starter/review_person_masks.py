#!/usr/bin/env python3
"""Review generated person masks with source, mask, and overlay thumbnails."""
import argparse,html,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,person_scene


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--out',required=True);p.add_argument('--per-page',type=int,default=5);a=p.parse_args()
 if not 1<=a.per_page<=10:raise ValueError('per-page must be 1–10')
 rows=json.loads(Path(a.manifest).read_text())
 if not isinstance(rows,list) or not rows:raise ValueError('Mask manifest is empty')
 inputs=[a.manifest];seen=set();entries=[]
 for r in rows:
  source=str(Path(r['source']).resolve());mask=str(Path(r['mask']).resolve())
  if source in seen:raise ValueError('Duplicate source')
  seen.add(source)
  if morph.sha256_file(source)!=r['source_sha256'] or morph.sha256_file(mask)!=r['mask_sha256']:raise ValueError('Source or mask changed since manifest')
  inputs.extend([source,mask]);entries.append((source,mask))
 out=Path(a.out);pages=math.ceil(len(entries)/a.per_page);outputs=[out/'index.html',out/'index.json']+[out/f'comparison-{i+1:02d}.jpg' for i in range(pages)];morph.protect_inputs(inputs,outputs);morph.protect_existing(outputs,False)
 if out.exists():raise ValueError('Output directory already exists')
 out.mkdir(parents=True);results=[];parts=['<!doctype html><meta charset="utf-8"><title>Person mask review</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;background:#f5f5f3;color:#222}article{background:white;padding:16px;margin:20px 0;border-radius:8px}img{max-width:100%;height:auto}</style><h1>Person mask review</h1><p>Thumbnails show the original, the grayscale mask, and a red overlay. Inspect each person and the surrounding edges. Masks can miss people or include background; an empty-mask flag does not assess mask quality.</p>']
 for start in range(0,len(entries),a.per_page):
  subset=entries[start:start+a.per_page];sheet=Image.new('RGB',(960,265*len(subset)),'white');draw=ImageDraw.Draw(sheet)
  for i,(source,mask) in enumerate(subset):
   x=morph.load(source,300);m=person_scene.mask_array(mask,x.shape);overlay=np.clip(x*(1-.5*m[...,None])+np.array([1.,.12,.08])*.5*m[...,None],0,1);gray=np.repeat(m[...,None],3,axis=-1);fraction=float(m.mean());flag=fraction<.001
   draw.text((8,i*265+5),f'{Path(source).name}   original | person mask | overlay'+('  [empty mask: review]' if flag else ''),fill='black')
   for j,array in enumerate([x,gray,overlay]):
    image=Image.fromarray(np.uint8(np.clip(array*255+.5,0,255)));image.thumbnail((300,235));sheet.paste(image,(j*320+10+(300-image.width)//2,i*265+24))
   results.append({'source':source,'mask':mask,'mean_mask_weight':fraction,'empty_mask_flag':flag,'sheet':f'comparison-{start//a.per_page+1:02d}.jpg'})
  name=f'comparison-{start//a.per_page+1:02d}.jpg';sheet.save(out/name,quality=92);parts.append(f'<article><h2>Photos {start+1}–{start+len(subset)}</h2><img src="{html.escape(name,quote=True)}" alt="Original, person mask, and overlay"></article>')
 parts.append('</html>');(out/'index.html').write_text(''.join(parts),encoding='utf-8');morph.dump({'scope':'Reduced-resolution mask overlays for human review. Blank-mask flag is a heuristic and does not validate mask coverage. Source and masks were checked against the manifest at review time.','photos':results},out/'index.json');print(len(results),'photos;',sum(x['empty_mask_flag'] for x in results),'empty mask flags')
if __name__=='__main__':main()
