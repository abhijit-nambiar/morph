#!/usr/bin/env python3
"""Compare one vs several edited representatives on held-out real pairs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import morph,gamut_renderer,joint_fit_experiment

def main():
    p=argparse.ArgumentParser();p.add_argument('groups');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--anchors',type=int,default=3);a=p.parse_args()
    if a.anchors<1:p.error('--anchors must be positive')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));edits=Path(a.edits);cases=[];visual=[]
    for g in groups:
        eligible=[]
        for path in g['photos']:
            target=edits/Path(path).name
            if not target.is_file():continue
            with Image.open(path) as x,Image.open(target) as y:
                if morph.oriented_size(x)==morph.oriented_size(y):eligible.append(path)
        if len(eligible)<2:continue
        rep=g['representative'] if g['representative'] in eligible else eligible[0]; max_anchors=min(a.anchors,len(eligible)-1)
        selected=joint_fit_experiment.choose_anchors({'representative':rep,'photos':eligible},set(),max_anchors,a.size);anchorpaths=[x[0] for x in selected];held=[p for p in eligible if p not in anchorpaths]
        im=morph.load(rep,a.size);target=morph.load(edits/Path(rep).name,a.size);single,sm=gamut_renderer.fit(im,target)
        imgs=[];targets=[];rng=np.random.default_rng(99)
        for path,x in selected:
            y=morph.load(edits/Path(path).name,a.size);xp=x.reshape(-1,3);yp=y.reshape(-1,3);idx=rng.choice(len(xp),min(3000,len(xp)),replace=False);imgs.append(xp[idx]);targets.append(yp[idx])
        joint,jm=gamut_renderer.fit(np.concatenate(imgs),np.concatenate(targets));results=[]
        for path in held:
            x=morph.load(path,a.size);y=morph.load(edits/Path(path).name,a.size);one=gamut_renderer.render(x,single);multi=gamut_renderer.render(x,joint);error=lambda z:float(np.sqrt(np.mean((z-y)**2)));row={'photo':path,'single_rmse':error(one),'joint_rmse':error(multi),'original_rmse':error(x)};results.append(row)
            visual.append((Path(path).name+' '+g['id'],(x,y,one,multi)))
        cases.append({'scene':g['id'],'representative':rep,'anchors':anchorpaths,'single_fit_metrics':sm,'joint_fit_metrics':jm,'single_recipe':gamut_renderer.recipe(single),'joint_recipe':gamut_renderer.recipe(joint),'heldout':results});print(g['id'],len(anchorpaths),'anchors',len(held),'heldout',flush=True)
    allrows=[v for c in cases for v in c['heldout']]
    if not allrows:raise ValueError('No groups have enough same-size matching pairs for held-out evaluation')
    x=np.array([r['single_rmse'] for r in allrows]);y=np.array([r['joint_rmse'] for r in allrows]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # Four columns: source, human edit, single fit transfer, multi fit transfer.
    from PIL import ImageDraw
    pages=[]
    for first in range(0,len(visual),10):
        subset=visual[first:first+10];sheet=Image.new('RGB',(800,175*len(subset)),'white');d=ImageDraw.Draw(sheet)
        for i,(label,images) in enumerate(subset):
            d.text((4,175*i+2),label+' | source / edited / one anchor / multiple anchors',fill='black')
            for col,arr in enumerate(images):
                image=Image.fromarray(np.uint8(np.clip(arr*255+.5,0,255)));image.thumbnail((192,150));sheet.paste(image,(col*200+4+(192-image.width)//2,175*i+20))
        name=f'comparison-{first//10+1:02d}.jpg';sheet.save(out/name,quality=90);pages.append(name)
    morph.dump({'scope':'One versus up to three diverse same-size original/human-edit anchors per proposed group, evaluated on different same-size photos. Anchors chosen from within the same album; no unseen-album or aesthetic validation. Same dimensions do not guarantee registration. All exports remain uncropped.','summary':{'groups':len(cases),'heldout_photos':len(allrows),'single_mean_rmse':float(x.mean()),'joint_mean_rmse':float(y.mean()),'single_median_rmse':float(np.median(x)),'joint_median_rmse':float(np.median(y)),'joint_improved':int(np.sum(y<x))},'cases':cases,'comparison_sheets':pages},out/'report.json')
if __name__=='__main__':main()
