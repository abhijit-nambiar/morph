#!/usr/bin/env python3
"""Compare one shared album mapping vs separate scene recipes on real edited pairs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('scene_report');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args();data=json.loads(Path(a.scene_report).read_text());scenes=[s for s in data['scenes'] if 'recipe' in s and s['heldout']];rng=np.random.default_rng(123);xs=[];ys=[]
    for scene in scenes:
        src=scene['representative'];img=morph.load(src,a.size);target=morph.load(Path(a.edits)/Path(src).name,a.size);x=img.reshape(-1,3);y=target.reshape(-1,3);idx=rng.choice(len(x),min(3000,len(x)),replace=False);xs.append(x[idx]);ys.append(y[idx])
    shared,metrics=gamut_renderer.fit(np.concatenate(xs),np.concatenate(ys));rows=[];visual=[]
    for scene in scenes:
        own=gamut_renderer.vec(scene['recipe'])
        for item in scene['heldout']:
            src=item['photo'];x=morph.load(src,a.size);y=morph.load(Path(a.edits)/Path(src).name,a.size);scene_img=gamut_renderer.render(x,own);album_img=gamut_renderer.render(x,shared);rmse=lambda image:float(np.sqrt(np.mean((image-y)**2)))
            rows.append({'scene':scene['id'],'photo':src,'original_rmse':rmse(x),'album_rmse':rmse(album_img),'scene_rmse':rmse(scene_img)});visual.append((Path(src).name, x,y,album_img,scene_img))
        print(scene['id'],len(scene['heldout']),'heldout',flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pages=[]
    for first in range(0,len(visual),10):
        subset=visual[first:first+10];sheet=Image.new('RGB',(800,175*len(subset)),'white');d=ImageDraw.Draw(sheet)
        for i,(name,*images) in enumerate(subset):
            d.text((4,i*175+2),name+' | original / supplied edit / album fit / scene fit',fill='black')
            for col,img in enumerate(images):
                im=Image.fromarray(np.uint8(np.clip(img*255+.5,0,255)));im.thumbnail((192,150));sheet.paste(im,(200*col+4+(192-im.width)//2,i*175+20))
        name=f'comparison-{first//10+1:02d}.jpg';sheet.save(out/name,quality=90);pages.append(name)
    baseline=np.array([r['original_rmse'] for r in rows]);album=np.array([r['album_rmse'] for r in rows]);scene=np.array([r['scene_rmse'] for r in rows]);morph.dump({'scope':'Same-sized pairs from one album. A shared album recipe is fit jointly on each group representative; each scene recipe was fit only on that group representative. The same heldout photos and renderer are used. Pixel error is not a photographer quality rating; grouping and alignment are not validated.','summary':{'anchors':len(scenes),'heldout':len(rows),'album_better_than_original':int(np.sum(album<baseline)),'scene_better_than_original':int(np.sum(scene<baseline)),'scene_better_than_album':int(np.sum(scene<album)),'original_mean_rmse':float(baseline.mean()),'album_mean_rmse':float(album.mean()),'scene_mean_rmse':float(scene.mean()),'original_median_rmse':float(np.median(baseline)),'album_median_rmse':float(np.median(album)),'scene_median_rmse':float(np.median(scene))},'album_recipe':gamut_renderer.recipe(shared),'album_fit_metrics':metrics,'heldout':rows,'comparison_sheets':pages},out/'report.json')
if __name__=='__main__':main()
