#!/usr/bin/env python3
"""Run a repeatable synthetic reconstruction and scene-transfer benchmark."""
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import morph

def main():
    p=argparse.ArgumentParser(); p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--heldout',type=int,default=3);p.add_argument('--size',type=int,default=320);p.add_argument('--engine',choices=['same','pillow'],default='same');a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text())); out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]; sheets=[]
    for group in groups:
        rep=group['representative']; x=morph.load(rep,a.size)
        held=[s for s in group['photos'] if s!=rep][:a.heldout]
        for name in ('warm','cool','soft'):
            known=morph.vec(morph.PRESETS[name]); target=morph.render(x,known) if a.engine=='same' else morph.pillow_target(x,name); fitted,metric=morph.fit(x,target)
            reconstructed=morph.render(x,fitted); full_rmse=float(np.sqrt(np.mean((target-reconstructed)**2)))
            others=[]
            for src in held:
                im=morph.load(src,a.size);want=morph.render(im,known) if a.engine=='same' else morph.pillow_target(im,name);got=morph.render(im,fitted)
                others.append({'photo':src,'rmse':float(np.sqrt(np.mean((want-got)**2)))})
            row={'scene':group['id'],'look':name,'representative':rep,'known_recipe':morph.recipe(known) if a.engine=='same' else None,'fitted_recipe':morph.recipe(fitted),'sampled_fit':metric,'representative_full_image_rmse':full_rmse,'heldout_photo_results':others}; rows.append(row)
            thumbs=[]
            for image in (x,target,reconstructed):
                im=Image.fromarray(np.uint8(np.clip(image*255+.5,0,255)));im.thumbnail((200,150));thumbs.append(im)
            sheet=Image.new('RGB',(620,190),'white'); draw=ImageDraw.Draw(sheet);draw.text((8,5),f"{group['id']} / {name}: original | target | reconstruction",fill='black')
            for i,im in enumerate(thumbs):sheet.paste(im,(10+200*i,28))
            sheets.append(sheet)
            print(group['id'],name,'representative RMSE',round(full_rmse,5),'heldout',len(others),flush=True)
    if sheets:
        sheet=Image.new('RGB',(620,190*len(sheets)),'white')
        for i,im in enumerate(sheets):sheet.paste(im,(0,190*i))
        sheet.save(out/'comparison.jpg',quality=90)
    vals=[y['rmse'] for x in rows for y in x['heldout_photo_results']]
    scope='Synthetic targets use the same renderer as fitting.' if a.engine=='same' else 'Synthetic targets use Pillow brightness, contrast and color, plus channel gains; this independent deterministic renderer is not a human edit.'
    morph.dump({'scope':scope,'engine':a.engine,'cases':rows,'summary':{'fit_cases':len(rows),'heldout_photo_look_cases':len(vals),'mean_heldout_rmse':float(np.mean(vals)) if vals else None,'max_heldout_rmse':max(vals) if vals else None}},out/'report.json')
if __name__=='__main__':main()
