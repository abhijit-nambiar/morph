#!/usr/bin/env python3
"""Inspect extrapolation, grayscale, and out-of-gamut behavior of fitted RGB mappings."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,poly_color

def audit(weights,steps=21):
    axis=np.linspace(0,1,steps,dtype=np.float32);r,g,b=np.meshgrid(axis,axis,axis,indexing='ij');colors=np.stack((r,g,b),axis=-1).reshape(-1,3);raw=colors+poly_color.features(colors)@weights
    low=raw<0;high=raw>1;clipped=poly_color.render(colors,weights);lum=clipped@np.array([.2126,.7152,.0722]);gray=np.linspace(0,1,256,dtype=np.float32);grayrgb=np.repeat(gray[:,None],3,axis=1);gray_out=poly_color.render(grayrgb,weights);gray_luma=gray_out@np.array([.2126,.7152,.0722]);return {'sampled_colors':len(colors),'out_of_gamut_fraction':float(np.mean(np.any(low|high,axis=1))),'below_zero_fraction':float(np.mean(np.any(low,axis=1))),'above_one_fraction':float(np.mean(np.any(high,axis=1))),'gray_luminance_reversal_steps':int(np.sum(np.diff(gray_luma)<-1e-6)),'black_output':gray_out[0].tolist(),'white_output':gray_out[-1].tolist()},grayrgb,gray_out

def main():
    p=argparse.ArgumentParser();p.add_argument('comparison');p.add_argument('--out',required=True);a=p.parse_args();data=json.loads(Path(a.comparison).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];sheet=Image.new('RGB',(800,90*len(data['scenes'])),'white');d=ImageDraw.Draw(sheet)
    for i,s in enumerate(data['scenes']):
        w=poly_color.deserialize(s['mapping']);report,before,after=audit(w);heldout=[]
        for photo in s.get('heldout',[]):
            image=morph.load(photo['photo'],256);colors=image.reshape(-1,3);raw=colors+poly_color.features(colors)@w
            heldout.append({'photo':photo['photo'],'out_of_gamut_fraction':float(np.mean(np.any((raw<0)|(raw>1),axis=1)))})
        report['heldout_mean_out_of_gamut_fraction']=float(np.mean([r['out_of_gamut_fraction'] for r in heldout])) if heldout else None
        rows.append({'scene':s['id'],'representative':s['representative'],'audit':report,'heldout':heldout});d.text((8,i*90+3),s['id']+' | neutral ramp original / output',fill='black')
        for j,arr in enumerate((before,after)):
            bar=np.tile(np.uint8(np.clip(arr*255+.5,0,255))[None,:,:],(25,1,1));im=Image.fromarray(bar).resize((760,25),Image.Resampling.NEAREST);sheet.paste(im,(10,i*90+20+j*28))
        print(s['id'],'outside',round(report['out_of_gamut_fraction'],3),'gray reversals',report['gray_luminance_reversal_steps'],flush=True)
    sheet.save(out/'neutral-ramps.jpg',quality=95);morph.dump({'scope':'Uniform RGB-cube samples are not the image color distribution; heldout estimates use resized photos before output encoding. Out-of-gamut means the unclipped mapping exceeds channel bounds; exported output is clipped. Gray reversal only checks monotonic luminance along the neutral axis, not all hue paths or perceptual quality.','rows':rows},out/'report.json')
if __name__=='__main__':main()
