#!/usr/bin/env python3
"""Compare near-limit statistics of two renderers using the same parameter labels."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,selective,gamut_renderer as gamut

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('recipe');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=512);a=p.parse_args();groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));params=selective.vec(json.loads(Path(a.recipe).read_text()));rows=[];vis=[]
    for g in groups:
        for src in g['photos']:
            x=morph.load(src,a.size);base=morph.clipping_diagnostic(x,params,renderer=selective.render);safe=morph.clipping_diagnostic(x,params,renderer=gamut.render);rows.append({'scene':g['id'],'photo':src,'current':base,'gamut':safe})
            if len(vis)<12:vis.append((Path(src).name,x,selective.render(x,params),gamut.render(x,params)))
        print(g['id'],len(g['photos']),flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);sheet=Image.new('RGB',(630,len(vis)*175),'white');d=ImageDraw.Draw(sheet)
    for i,(name,*imgs) in enumerate(vis):
        d.text((3,175*i+2),name+'   original / current / gamut renderer',fill='black')
        for j,arr in enumerate(imgs):
            im=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));im.thumbnail((202,150));sheet.paste(im,(210*j+4+(202-im.width)//2,175*i+20))
    sheet.save(out/'comparison.jpg',quality=90)
    s={};
    for key in ('current','gamut'):
        s[key]={'photos_flagged':sum(r[key]['flag'] for r in rows),'median_high_increase':float(np.median([r[key]['high_increase'] for r in rows])),'median_low_increase':float(np.median([r[key]['low_increase'] for r in rows]))}
    morph.dump({'scope':'Experimental renderers use different parameter semantics; the same numeric recipe produces different looks. Near-limit sampling is a heuristic, not a measure of preserved detail or artistic quality.','summary':s,'rows':rows},out/'report.json')
if __name__=='__main__':main()
