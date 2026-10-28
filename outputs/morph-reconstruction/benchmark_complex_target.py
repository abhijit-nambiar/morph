#!/usr/bin/env python3
"""Benchmark independent global + HSV-selective target generation and transfer."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,selective,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--heldout',type=int,default=3);p.add_argument('--size',type=int,default=320);p.add_argument('--target',choices=['pillow-selective','pillow-soft'],default='pillow-selective');a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));out=Path(a.out);out.mkdir(parents=True,exist_ok=True);records=[];vis=[]
    make_target=morph.pillow_soft_target if a.target=='pillow-soft' else morph.pillow_selective_target
    for g in groups:
        x=morph.load(g['representative'],a.size);others=[s for s in g['photos'] if s!=g['representative']][:a.heldout];others=[(s,morph.load(s,a.size)) for s in others]
        for look in ('warm','cool','soft'):
            y=make_target(x,look);fits={};results={}
            for name in ('global','selective','gamut'):
                if name=='global':p_,m=morph.run_fit(x,y,method='gauss-newton');render=morph.render;to_recipe=morph.recipe
                else:
                    model=selective if name=='selective' else gamut_renderer;p_,m=model.fit(x,y);render=model.render;to_recipe=model.recipe
                fits[name]={'recipe':to_recipe(p_),'fit_metrics':m};results[name]=[]
                for filename,held in others:
                    target=make_target(held,look);reconstructed=render(held,p_);diagnostic=morph.clipping_diagnostic(held,p_,renderer=render)
                    target_high=float(np.mean(np.any(target>=254/255,axis=-1)));target_low=float(np.mean(np.all(target<=1/255,axis=-1)))
                    results[name].append({'photo':filename,'rmse':float(np.sqrt(np.mean((target-reconstructed)**2))),'clipping':diagnostic,'target_near_white_fraction':target_high,'target_near_black_fraction':target_low})
            records.append({'scene':g['id'],'look':look,'representative':g['representative'],'fits':fits,'heldout':results});
            if len(vis)<12:vis.append((g['id']+' '+look,(x,y,morph.render(x,morph.vec(fits['global']['recipe'])),selective.render(x,selective.vec(fits['selective']['recipe'])),gamut_renderer.render(x,gamut_renderer.vec(fits['gamut']['recipe'])))))
            print(g['id'],look,flush=True)
    sheet=Image.new('RGB',(1000,175*len(vis)),'white');d=ImageDraw.Draw(sheet)
    for i,(label,imgs) in enumerate(vis):
        d.text((5,i*175+2),label+'    original / target / global fit / selective fit / gamut fit',fill='black')
        for j,arr in enumerate(imgs):
            im=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));im.thumbnail((192,150));sheet.paste(im,(j*200+3+(192-im.width)//2,i*175+20))
    sheet.save(out/'comparison.jpg',quality=90)
    summary={}
    for name in ('global','selective','gamut'):
        rows=[r for c in records for r in c['heldout'][name]];values=[r['rmse'] for r in rows];summary[name]={'heldout':len(values),'mean_rmse':float(np.mean(values)),'median_rmse':float(np.median(values)),'clipping_flags':sum(r['clipping']['flag'] for r in rows),'mean_near_white_increase':float(np.mean([r['clipping']['high_increase'] for r in rows])),'mean_target_near_white_fraction':float(np.mean([r['target_near_white_fraction'] for r in rows])),'mean_output_near_white_fraction':float(np.mean([r['clipping']['after']['high'] for r in rows]))}
    morph.dump({'scope':'Independent Pillow-based tone operations and HSV hue-range adjustments; no geometry changes. Representative fit and heldout transfer are tested on the same album and unvalidated proposed groups.','target_engine':a.target,'summary':summary,'cases':records},out/'report.json')
if __name__=='__main__':main()
