#!/usr/bin/env python3
"""Probe feature distances after hiding segmented people in one proposed scene."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import morph

groups=json.load(open('outputs/morph-reconstruction/groups-chronological.json'));scene=next(g for g in groups['groups'] if g['id']=='scene-02');paths=[str(Path(p).resolve()) for p in scene['photos']];root=Path('outputs/morph-reconstruction/segmentation-scene-02');root.mkdir(exist_ok=False);hashes={p:morph.sha256_file(p) for p in paths};conditioned=[]
for path in paths:
 name=Path(path).stem;mask=root/(name+'-mask.png');subprocess.run(['outputs/morph-reconstruction/person-segmentation',path,str(mask)],check=True,stdout=subprocess.DEVNULL)
 x=morph.load(path,1024);m=Image.open(mask).convert('L').resize((x.shape[1],x.shape[0]),Image.Resampling.BILINEAR);alpha=np.asarray(m,dtype=np.float32)[...,None]/255;conditioned_image=np.uint8(np.clip(x*(1-alpha)+.5*alpha,0,1)*255+.5);out=root/(name+'-background.png');Image.fromarray(conditioned_image).save(out);conditioned.append(str(out.resolve()));print(name,flush=True)
if any(morph.sha256_file(p)!=h for p,h in hashes.items()):raise ValueError('Original changed')
(root/'conditioned-paths.json').write_text(json.dumps(conditioned));result=subprocess.run(['outputs/morph-reconstruction/vision-scene-features',str(root/'conditioned-paths.json')],check=True,capture_output=True);data=json.loads(result.stdout);data['source_paths']=paths;data['source_sha256']=hashes;data['scope']='Exploratory whole-frame feature distances after replacing predicted person-mask pixels with neutral gray; masks may miss people and cover background, no inpainting, not a scene embedding. Original orientation and background image are preserved.';morph.dump(data,root/'background-distances.json')
