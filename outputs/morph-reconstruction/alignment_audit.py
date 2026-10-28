#!/usr/bin/env python3
"""Flag obvious translational misalignment using gradient correlation on thumbnails."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import morph

def edge(x):
    y=x@np.array([.2126,.7152,.0722],dtype=np.float32)
    dy,dx=np.gradient(y);return np.stack([dx,dy],axis=-1)

def alignment(x,y,maxshift=8):
    if x.shape!=y.shape:raise ValueError('requires matching dimensions')
    a=edge(x); b=edge(y);h,w=a.shape[:2]
    def score(dx,dy):
        ax0=max(0,-dx);ay0=max(0,-dy); ax1=min(w,w-dx);ay1=min(h,h-dy)
        u=a[ay0:ay1,ax0:ax1];v=b[ay0+dy:ay1+dy,ax0+dx:ax1+dx]
        return float(np.sum(u*v)/max(np.sqrt(np.sum(u*u)*np.sum(v*v)),1e-12))
    zero=score(0,0); best=(-2.,0,0)
    for dy in range(-maxshift,maxshift+1):
        for dx in range(-maxshift,maxshift+1):
            s=score(dx,dy)
            if s>best[0]:best=(s,dx,dy)
    improvement=best[0]-zero
    return {'zero_score':zero,'best_score':best[0],'dx':best[1],'dy':best[2],'score_improvement':improvement,'flag':(max(abs(best[1]),abs(best[2]))>=2 and improvement>=.02) or zero<.4}

def main():
    p=argparse.ArgumentParser();p.add_argument('originals');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=128);p.add_argument('--maxshift',type=int,default=8);a=p.parse_args();rows=[]
    for src in morph.files(a.originals):
        dst=Path(a.edits)/src.name
        if not dst.is_file():continue
        with Image.open(src) as x,Image.open(dst) as y:
            if morph.oriented_size(x)!=morph.oriented_size(y): rows.append({'file':src.name,'status':'different dimensions'});continue
        x=morph.load(src,a.size);y=morph.load(dst,a.size);r=alignment(x,y,a.maxshift);rows.append({'file':src.name,'status':'audited','analysis':r});print(src.name,'flag',r['flag'],'offset',r['dx'],r['dy'],flush=True)
    morph.dump({'method':'Gradient-correlation thumbnail translation search. Heuristic only: correlation is sensitive to content, color processing, scale and rotation. Flag does not prove a crop, and an unflagged pair is not guaranteed aligned. dx/dy are offsets in thumbnail pixels.','size':a.size,'rows':rows},a.out)
if __name__=='__main__':main()
