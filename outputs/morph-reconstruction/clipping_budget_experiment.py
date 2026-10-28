#!/usr/bin/env python3
"""Explore per-photo recipe strength constrained by a heuristic clipping budget."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,selective

def main():
    p=argparse.ArgumentParser();p.add_argument('album_report');p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--budget',type=float,default=.01);p.add_argument('--size',type=int,default=512);a=p.parse_args()
    if not 0<=a.budget<=1:raise ValueError('budget must be 0..1')
    album=Path(a.album_report).parent; groups=morph.validate_groups(json.loads(Path(a.groups).read_text())); rows=[];examples=[]
    for g in groups:
        d=json.loads((album/g['id']/'fit.json').read_text()); renderer=selective.render if d.get('renderer')=='selective' else morph.render
        p=selective.vec(d['fitted_recipe']) if d.get('renderer')=='selective' else morph.vec(d['fitted_recipe'])
        for path in g['photos']:
            x=morph.load(path,a.size);chosen=0.;before=morph.clipping_diagnostic(x,p,renderer)
            for s in np.linspace(1,0,21):
                diag=morph.clipping_diagnostic(x,p*s,renderer)
                if diag['high_increase']<=a.budget and diag['low_increase']<=a.budget: chosen=float(s);break
            out=morph.render(x,p*chosen) if d.get('renderer')!='selective' else selective.render(x,p*chosen)
            rows.append({'scene':g['id'],'photo':path,'strength':chosen,'full':before,'budgeted':diag})
            if len(examples)<12:examples.append((Path(path).name,x,renderer(x,p),out,chosen))
        print(g['id'],len(g['photos']),flush=True)
    sheet=Image.new('RGB',(630,len(examples)*175),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,orig,full,budgeted,s) in enumerate(examples):
        draw.text((4,i*175+2),f'{name}: original / full / budgeted (strength {s:.2f})',fill='black')
        for j,arr in enumerate((orig,full,budgeted)):
            im=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));im.thumbnail((202,150));sheet.paste(im,(j*210+4+(202-im.width)//2,i*175+20))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);sheet.save(out/'comparison.jpg',quality=90)
    strengths=np.array([r['strength'] for r in rows]);morph.dump({'scope':'Heuristic search over 21 strengths at thumbnail resolution; does not guarantee full-resolution or encoded clipping bounds or consistent look. No aesthetic evaluation.','budget_fraction':a.budget,'summary':{'photos':len(rows),'median_strength':float(np.median(strengths)),'mean_strength':float(np.mean(strengths)),'zero_strength':int(sum(strengths==0)),'full_strength':int(sum(strengths==1))},'photos':rows},out/'report.json')
if __name__=='__main__':main()
