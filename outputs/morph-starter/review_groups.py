#!/usr/bin/env python3
"""Make contact sheets for reviewing proposed scenes and representative images."""
import argparse,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import morph
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--columns',type=int,default=5);p.add_argument('--source',help='Optional album folder to audit for unassigned photos');a=p.parse_args()
    if a.columns<1 or a.columns>10: p.error('--columns must be between 1 and 10')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    index=[]
    for g in groups:
        photos=g['photos']; rep=g['representative']; width=230; height=190
        sheet=Image.new('RGB',(a.columns*width,40+height*math.ceil(len(photos)/a.columns)),(247,247,245));d=ImageDraw.Draw(sheet)
        d.text((10,12),f"{g['id']} - {len(photos)} photos; representative marked with a red border",fill=(20,20,20))
        for i,path in enumerate(photos):
            x=width*(i%a.columns); y=40+height*(i//a.columns)
            image=Image.fromarray(np.uint8(morph.load(path,220)*255+.5)); image.thumbnail((210,150))
            sheet.paste(image,(x+10+(210-image.width)//2,y+(150-image.height)//2))
            if path==rep: d.rectangle((x+5,y-2,x+224,y+153),outline=(210,55,45),width=3)
            d.text((x+8,y+156),f"{i+1:02d} {Path(path).name}",fill=(15,15,15))
        filename=g['id']+'.jpg';sheet.save(out/filename,quality=90)
        index.append({'id':g['id'],'representative':rep,'photo_count':len(photos),'contact_sheet':filename})
        print(f"{g['id']}: {len(photos)} photos -> {out/filename}")
    assigned={str(Path(s).resolve()) for g in groups for s in g['photos']}
    source={str(s) for s in morph.files(a.source)} if a.source else None
    unassigned=sorted(source-assigned) if source is not None else None
    outside=sorted(assigned-source) if source is not None else None
    if unassigned or outside: print(f'Warning: {len(unassigned)} unassigned and {len(outside)} outside the source folder.')
    morph.dump({'groups':index,'unassigned':unassigned,'outside_source':outside},out/'index.json')
if __name__=='__main__':main()
