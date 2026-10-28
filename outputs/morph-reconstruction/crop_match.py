#!/usr/bin/env python3
"""Experimental scale/translation crop matching for evaluation, not image export."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph

def edges(x):
    y=x@np.array([.2126,.7152,.0722],dtype=np.float32); dy,dx=np.gradient(y);return dx,dy

def window_sum(arr,h,w):
    integral=np.pad(arr.cumsum(0).cumsum(1),((1,0),(1,0)))
    return integral[h:,w:]-integral[:-h,w:]-integral[h:,:-w]+integral[:-h,:-w]

def match(original,target,steps=31,max_scale=1.75,size=144):
    orig_h,orig_w=original.shape[:2]; targ_h,targ_w=target.shape[:2]
    target_im=Image.fromarray(np.uint8(target*255+.5));target_im.thumbnail((size,size)); t=np.asarray(target_im,dtype=np.float32)/255;th,tw=t.shape[:2]; tx,ty=edges(t);t_energy=float(np.sum(tx*tx+ty*ty))
    if t_energy<1e-7: raise ValueError('Target has too little texture to align')
    cover=max(tw/orig_w,th/orig_h); best=None
    orig_im=Image.fromarray(np.uint8(original*255+.5))
    for factor in np.linspace(1,max_scale,steps):
        sw=max(tw,int(round(orig_w*cover*factor)));sh=max(th,int(round(orig_h*cover*factor)))
        src=np.asarray(orig_im.resize((sw,sh),Image.Resampling.BILINEAR),dtype=np.float32)/255;sx,sy=edges(src)
        shape=(sh+th-1,sw+tw-1)
        corr=np.fft.irfft2(np.fft.rfft2(sx,shape)*np.fft.rfft2(tx[::-1,::-1],shape)+np.fft.rfft2(sy,shape)*np.fft.rfft2(ty[::-1,::-1],shape),shape)[th-1:sh,tw-1:sw]
        energy=window_sum(sx*sx+sy*sy,th,tw);score=corr/np.sqrt(np.maximum(energy*t_energy,1e-12))
        iy,ix=np.unravel_index(np.argmax(score),score.shape);value=float(score[iy,ix]);candidate={'score':value,'factor':float(factor),'box':[float(ix/sw),float(iy/sh),float((ix+tw)/sw),float((iy+th)/sh)]}
        if best is None or value>best['score']:best=candidate
    return best

def main():
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('target');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=144);a=p.parse_args();x=morph.load(a.original,1200);y=morph.load(a.target,1200);result=match(x,y,size=a.size);box=result['box'];h,w=x.shape[:2];coords=(int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h));crop=x[coords[1]:coords[3],coords[0]:coords[2]];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);morph.save(crop,out/'estimated-crop.jpg');morph.save(y,out/'target-preview.jpg');morph.dump({'original':str(Path(a.original).resolve()),'target':str(Path(a.target).resolve()),'estimate':result,'warning':'Exploratory crop estimate; scores are not calibrated. Does not estimate rotation or local warps. Inspect before using for evaluation; never use this crop for export.'},out/'match.json');print(json.dumps(result))
if __name__=='__main__':main()
