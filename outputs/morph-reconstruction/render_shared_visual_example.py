import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,poly_color

report=json.load(open('outputs/morph-reconstruction/shared-visual-group-evaluation.json'));rows=[r for r in report['photos'] if r['group']=='scene-15']
selected=[rows[i] for i in (0,2,4,6)];fixed=selected[0]['fixed_reference'];visual=selected[0]['visual_reference'];edits=Path('/Users/abhijitnambiar/Downloads/Morph/edits')
def fit(p):
 x=morph.load(p,report['size']);y=morph.load(edits/Path(p).name,report['size']);return poly_color.fit(x,y,regularization=report['regularization'])[0]
w1=fit(fixed);w2=fit(visual);sheet=Image.new('RGB',(1280,4*260+50),'white');draw=ImageDraw.Draw(sheet)
for j,label in enumerate(['Original','Supplied edit','Chronological reference','Visual group reference']):draw.text((j*320+10,12),label,fill='black')
for i,row in enumerate(selected):
 p=Path(row['photo']);x=morph.load(p,500);target=morph.load(edits/p.name,500);images=[x,target,poly_color.render(x,w1),poly_color.render(x,w2)]
 for j,arr in enumerate(images):
  image=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));image.thumbnail((300,225));sheet.paste(image,(j*320+10+(300-image.width)//2,50+i*260))
 draw.text((10,50+i*260+228),p.name,fill='black')
sheet.save('outputs/morph-reconstruction/shared-visual-group-example.jpg',quality=94)
