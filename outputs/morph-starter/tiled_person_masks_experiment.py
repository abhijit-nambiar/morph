#!/usr/bin/env python3
"""Experimental multi-scale person masks; windows are used only for detection."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import morph


def run(extractor,source,out):
 subprocess.run([extractor,str(source),str(out)],check=True,stdout=subprocess.DEVNULL)
 with Image.open(out) as image:
  if image.format!='PNG' or image.mode!='L':raise ValueError('Expected grayscale mask')
  return image.copy()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--extractor',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 rows=json.loads(Path(a.manifest).read_text());out=Path(a.out).resolve();extractor=str(Path(a.extractor).resolve())
 if not rows or out.exists() or not out.parent.is_dir():raise ValueError('Invalid manifest or output')
 for row in rows:
  if morph.sha256_file(row['source'])!=row['source_sha256'] or morph.sha256_file(row['mask'])!=row['mask_sha256']:raise ValueError('Source or mask changed')
 with tempfile.TemporaryDirectory(prefix='.multiscale-masks-',dir=out.parent) as td:
  stage=Path(td);work=stage/'work';work.mkdir();result=[]
  for row in rows:
   path=row['source'];x=morph.load(path,1600);h,w=x.shape[:2];canvas=Image.fromarray(np.uint8(x*255+.5));full=run(extractor,path,work/'full.png').resize((w,h),Image.Resampling.BILINEAR);(work/'full.png').unlink();combined=np.asarray(full,dtype=np.float32)/255
   tw=max(1,round(w*.65));th=max(1,round(h*.65))
   for left,top in [(0,0),(w-tw,0),(0,h-th),(w-tw,h-th)]:
    tile=work/'tile.png';mp=work/'tile-mask.png';canvas.crop((left,top,left+tw,top+th)).save(tile);mask=run(extractor,tile,mp).resize((tw,th),Image.Resampling.BILINEAR);tile.unlink();mp.unlink();arr=np.asarray(mask,dtype=np.float32)/255
    # Taper only interior window edges so artificial window boundaries do not dominate.
    dx=max(1,int(tw*.08));dy=max(1,int(th*.08));wx=np.ones(tw,dtype=np.float32);wy=np.ones(th,dtype=np.float32)
    if left>0:wx[:dx]=np.linspace(0,1,dx)
    if left+tw<w:wx[-dx:]=np.linspace(1,0,dx)
    if top>0:wy[:dy]=np.linspace(0,1,dy)
    if top+th<h:wy[-dy:]=np.linspace(1,0,dy)
    combined[top:top+th,left:left+tw]=np.maximum(combined[top:top+th,left:left+tw],arr*wy[:,None]*wx[None,:])
   name=Path(path).stem+'-mask.png';dest=stage/name;Image.fromarray(np.uint8(np.clip(combined*255+.5,0,255))).save(dest);result.append({'source':path,'source_sha256':row['source_sha256'],'mask':str(out/name),'mask_sha256':morph.sha256_file(dest)});print(Path(path).name,flush=True)
  if any(morph.sha256_file(r['source'])!=r['source_sha256'] or morph.sha256_file(r['mask'])!=r['mask_sha256'] for r in rows):raise ValueError('Inputs changed during segmentation')
  work.rmdir();morph.dump(result,stage/'manifest.json');os.rename(stage,out)
if __name__=='__main__':main()
