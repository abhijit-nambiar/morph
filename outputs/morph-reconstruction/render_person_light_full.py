import json
from pathlib import Path
import numpy as np
from PIL import Image
import morph,poly_color
report=json.load(open('outputs/morph-reconstruction/person-light-synthetic-experiment.json'));s=next(x for x in report['scenes'] if x['scene']=='scene-02-lawn');w=np.array(s['person_weights']);photo=next(x['photo'] for x in report['photos'] if x['photo'].endswith('DSC_0974-18.jpg'));x=morph.load(photo);mask_file=Path('outputs/morph-reconstruction/segmentation-scene-02')/(Path(photo).stem+'-mask.png');mask=np.asarray(Image.open(mask_file).convert('L').resize((x.shape[1],x.shape[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
class Renderer:
 def render_tile(self,tile,weights,start,shape):
  m=mask[start:start+tile.shape[0],:,None];features=np.concatenate([poly_color.features(tile),m*tile*(1-tile),m],axis=-1);return np.clip(tile+features@weights,0,1)
 def __call__(self,tile,weights):raise RuntimeError('Use tiled rendering')
out=Path('outputs/morph-reconstruction/person-light-full-resolution/DSC_0974-18.jpg');morph.protect_inputs([photo,mask_file],[out]);morph.protect_existing([out],False);morph.save_render(x,w,out,renderer=Renderer(),source=photo);print(out, x.shape)
