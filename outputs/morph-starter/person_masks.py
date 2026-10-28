#!/usr/bin/env python3
"""Generate inspectable person masks with an external local Vision helper."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
from PIL import Image
import morph


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('--scene',required=True);p.add_argument('--extractor',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));group=next((g for g in groups if g['id']==a.scene),None)
 if group is None:raise ValueError('Unknown scene')
 out=Path(a.out).resolve();extractor=Path(a.extractor).resolve()
 if out.exists() or not out.parent.is_dir() or not extractor.is_file():raise ValueError('Invalid output or extractor')
 paths=[str(Path(p).resolve()) for p in group['photos']];names=[Path(p).stem.lower() for p in paths]
 if len(set(names))!=len(names):raise ValueError('Mask output stems must be unique')
 before={p:morph.sha256_file(p) for p in paths}
 with tempfile.TemporaryDirectory(prefix='.person-masks-',dir=out.parent) as td:
  stage=Path(td);rows=[]
  for path in paths:
   name=Path(path).stem+'-mask.png';mask=stage/name;subprocess.run([str(extractor),path,str(mask)],check=True)
   with Image.open(mask) as im:
    if im.format!='PNG' or im.mode not in ('L','I;16'):raise ValueError('Extractor did not produce a grayscale PNG')
   rows.append({'source':path,'source_sha256':before[path],'mask':str(out/name),'mask_sha256':morph.sha256_file(mask)})
  if any(morph.sha256_file(p)!=h for p,h in before.items()):raise ValueError('Source changed during segmentation')
  morph.dump(rows,stage/'manifest.json');os.rename(stage,out)
 print('Generated',len(rows),'masks; inspect before fitting:',out)
if __name__=='__main__':main()
