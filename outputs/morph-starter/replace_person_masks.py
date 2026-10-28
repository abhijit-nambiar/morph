#!/usr/bin/env python3
"""Create a new checked mask manifest with explicitly selected replacement masks."""
import argparse,json,os,shutil,tempfile
from pathlib import Path
from PIL import Image
import morph


def checked_mask(path):
 with Image.open(path) as im:
  if im.format!='PNG' or im.mode!='L' or min(im.size)<8:raise ValueError('Mask must be an 8-bit grayscale PNG of at least 8 pixels per side: '+str(path))
  im.verify()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('--replace',nargs=2,action='append',metavar=('PHOTO_FILENAME','MASK_PNG'),required=True);p.add_argument('--out',required=True);a=p.parse_args()
 rows=json.loads(Path(a.manifest).read_text());out=Path(a.out).resolve()
 if not isinstance(rows,list) or not rows or out.exists() or not out.parent.is_dir():raise ValueError('Invalid manifest or output')
 lookup={};seen=set()
 for row in rows:
  src=str(Path(row['source']).resolve());mask=str(Path(row['mask']).resolve());name=Path(src).name
  if src in seen or name in lookup:raise ValueError('Duplicate source or filename')
  seen.add(src);lookup[name]=row
  if morph.sha256_file(src)!=row['source_sha256'] or morph.sha256_file(mask)!=row['mask_sha256']:raise ValueError('Original source or mask changed')
  checked_mask(mask)
 replacements={}
 for name,path in a.replace:
  if name not in lookup or name in replacements:raise ValueError('Unknown or duplicate replacement photo: '+name)
  path=str(Path(path).resolve());checked_mask(path);replacements[name]=path
 hashes={path:morph.sha256_file(path) for path in replacements.values()};names=[Path(r['source']).stem.lower() for r in rows]
 if len(names)!=len(set(names)):raise ValueError('Output mask names would collide')
 with tempfile.TemporaryDirectory(prefix='.reviewed-masks-',dir=out.parent) as td:
  stage=Path(td);result=[]
  for row in rows:
   name=Path(row['source']).name;source_mask=replacements.get(name,str(Path(row['mask']).resolve()));filename=Path(name).stem+'-mask.png';dest=stage/filename;shutil.copyfile(source_mask,dest)
   result.append({'source':str(Path(row['source']).resolve()),'source_sha256':row['source_sha256'],'mask':str(out/filename),'mask_sha256':morph.sha256_file(dest)})
  if any(morph.sha256_file(r['source'])!=r['source_sha256'] or morph.sha256_file(r['mask'])!=r['mask_sha256'] for r in rows) or any(morph.sha256_file(p)!=h for p,h in hashes.items()):raise ValueError('Input changed during replacement')
  morph.dump(result,stage/'manifest.json');os.rename(stage,out)
 print('Created',len(result),'masks with',len(replacements),'explicit replacements; review alignment before fitting')
if __name__=='__main__':main()
