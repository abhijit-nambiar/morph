#!/usr/bin/env python3
"""Numerically sanity-check supported prompt directions on scene representatives."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import morph,prompt_target,poly_color,gamut_renderer

def stats(img,neutral):
    y=img@np.array([.2126,.7152,.0722],dtype=np.float32);base=neutral@np.array([.2126,.7152,.0722],dtype=np.float32)
    shadow=base<=np.quantile(base,.2);highlight=base>=np.quantile(base,.8)
    raw=Image.fromarray(np.uint8(np.clip(neutral*255+.5,0,255))).convert('HSV');hsv0=np.asarray(raw,dtype=np.float32)/255
    hsv=np.asarray(Image.fromarray(np.uint8(np.clip(img*255+.5,0,255))).convert('HSV'),dtype=np.float32)/255
    hue=hsv0[...,0];sat=hsv0[...,1];blue=(abs(hue-.62)<.12)&(sat>.12);warm=(np.minimum(abs(hue-.08),1-abs(hue-.08))<.10)&(sat>.12)
    return {'red_minus_blue':float(np.mean(img[...,0]-img[...,2])),'mean_luma':float(y.mean()),'contrast_luma':float(y.std()),'shadows_luma':float(y[shadow].mean()),'highlights_luma':float(y[highlight].mean()),'blue_saturation':float(hsv[...,1][blue].mean()) if np.count_nonzero(blue)>100 else None,'warm_saturation':float(hsv[...,1][warm].mean()) if np.count_nonzero(warm)>100 else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=320);p.add_argument('--renderer',choices=['polynomial','gamut'],default='polynomial');a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));cases=[]
    expected={'temperature':('red_minus_blue',{'warm':1,'cool':-1}),'exposure':('mean_luma',{'brighter':1,'darker':-1}),'contrast':('contrast_luma',{'soft contrast':-1,'strong contrast':1}),'shadows':('shadows_luma',{'lift shadows':1,'deepen shadows':-1}),'highlights':('highlights_luma',{'protect highlights':-1,'brighter highlights':1}),'blue':('blue_saturation',{'muted blues':-1,'vivid blues':1}),'warmcolors':('warm_saturation',{'muted warm colors':-1,'vivid warm colors':1})}
    for g in groups:
        x=morph.load(g['representative'],a.size);neutral=morph.pillow_soft_target(x,'neutral',{});base=stats(neutral,neutral)
        held=next((path for path in g['photos'] if path!=g['representative']),None)
        if held:
            held_image=morph.load(held,a.size);held_neutral=morph.pillow_soft_target(held_image,'neutral',{});held_base=stats(held_neutral,held_neutral)
        for phrase,(category,_) in prompt_target.PHRASES.items():
            settings,_=prompt_target.parse(phrase);target=morph.pillow_soft_target(x,'neutral',settings);actual=stats(target,neutral);metric,signs=expected[category];before=base[metric];after=actual[metric];delta=None if before is None or after is None else after-before;status='not measurable' if delta is None else ('expected direction' if delta*signs[phrase]>1e-6 else ('negligible' if abs(delta)<=1e-6 else 'opposite direction'))
            transfer_status='not measurable';transfer_delta=None
            if held:
                if a.renderer=='polynomial':fitted,_=poly_color.fit(x,target,regularization=.001);transferred=poly_color.render(held_image,fitted)
                else:fitted,_=gamut_renderer.fit(x,target);transferred=gamut_renderer.render(held_image,fitted)
                transferred_stats=stats(transferred,held_neutral);b=held_base[metric];v=transferred_stats[metric]
                transfer_delta=None if b is None or v is None else v-b
                transfer_status='not measurable' if transfer_delta is None else ('expected direction' if transfer_delta*signs[phrase]>1e-6 else ('negligible' if abs(transfer_delta)<=1e-6 else 'opposite direction'))
            cases.append({'scene':g['id'],'representative':g['representative'],'heldout_photo':held,'phrase':phrase,'metric':metric,'delta':delta,'status':status,'transfer_delta':transfer_delta,'transfer_status':transfer_status})
        print(g['id'],flush=True)
    summary={};transfer_summary={}
    for phrase in prompt_target.PHRASES:
        r=[x for x in cases if x['phrase']==phrase];summary[phrase]={s:sum(x['status']==s for x in r) for s in ('expected direction','opposite direction','negligible','not measurable')};transfer_summary[phrase]={s:sum(x['transfer_status']==s for x in r) for s in ('expected direction','opposite direction','negligible','not measurable')}
    morph.dump({'scope':'Numerical directional sanity checks on resized representatives and one different photo per group. Saturation uses simple hue masks. No semantic understanding, visual-quality assessment, or unseen-album validation.','renderer':a.renderer,'summary':summary,'transfer_summary':transfer_summary,'cases':cases},a.out)
if __name__=='__main__':main()
