#!/usr/bin/env python3
"""Research-only synthetic face-relative lighting transfer using offline face boxes."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import morph,poly_color,grid_light


def mask(shape,faces):
    h,w=shape[:2];yy=(np.arange(h,dtype=np.float32)+.5)/h;xx=(np.arange(w,dtype=np.float32)+.5)/w;result=np.zeros((h,w),dtype=np.float32)
    for face in faces:
        cx=face['x']+face['width']/2;cy=face['y']+face['height']/2
        sx=max(face['width']*.8,.025);sy=max(face['height']*.8,.025)
        m=np.exp(-.5*(((xx[None,:]-cx)/sx)**2+((yy[:,None]-cy)/sy)**2));result=np.maximum(result,m)
    return result

def features(x,faces):return np.concatenate([poly_color.features(x),mask(x.shape,faces)[...,None]*x*(1-x)],axis=-1)
def fit(x,y,faces,penalty=.0001):
    rng=np.random.default_rng(0);idx=rng.choice(x.shape[0]*x.shape[1],min(x.shape[0]*x.shape[1],30000),replace=False)
    a=features(x,faces).reshape(-1,13)[idx].astype(np.float64);b=(y-x).reshape(-1,3)[idx].astype(np.float64)
    return np.linalg.lstsq(a.T@a/len(a)+penalty*np.eye(13),a.T@b/len(a),rcond=None)[0]
def render(x,faces,weights):return np.clip(x+features(x,faces)@weights,0,1)
def target(x,faces,settings,strength):return np.clip(morph.pillow_soft_target(x,'neutral',settings)+strength*mask(x.shape,faces)[...,None]*x*(1-x),0,1)
def rmse(x,y):return float(np.sqrt(np.mean((x-y)**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('detections');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));detections=json.loads(Path(a.detections).read_text());lookup={str(Path(r['path']).resolve()):r for r in detections};settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[];pages=[];panels=[]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for g in groups:
        rep=g['representative'];r=lookup[str(Path(rep).resolve())]
        if r.get('error') or not r['faces']:continue
        original=morph.load(rep,a.size);held=[p for p in g['photos'] if p!=rep]
        for strength in (0.,.3):
            reference=target(original,r['faces'],settings,strength);wg,_=grid_light.fit_with_metrics(original,reference,regularization=.0001);wf=fit(original,reference,r['faces'])
            for path in held:
                d=lookup[str(Path(path).resolve())]
                if d.get('error') or not d['faces']:continue
                x=morph.load(path,a.size);wanted=target(x,d['faces'],settings,strength);gr=grid_light.render(x,wg);face=render(x,d['faces'],wf)
                rows.append({'scene':g['id'],'representative':str(rep),'photo':str(path),'faces':len(d['faces']),'strength':strength,'grid_rmse':rmse(gr,wanted),'face_rmse':rmse(face,wanted),'source_rmse':rmse(x,wanted)})
                if strength==.3:panels.append((Path(path).name,x,d['faces']))
        print(g['id'],len(held),'held-out photos',flush=True)
    if not rows:raise ValueError('No usable detection pairs')
    for start in range(0,len(panels),8):
        subset=panels[start:start+8];sheet=Image.new('RGB',(600,200*len(subset)),'white');draw=ImageDraw.Draw(sheet)
        for i,(name,x,faces) in enumerate(subset):
            im=Image.fromarray(np.uint8(x*255+.5));im.thumbnail((285,170));px=7+(285-im.width)//2;py=200*i+22;sheet.paste(im,(px,py));draw.text((5,200*i+3),name+' | detected faces / target light mask',fill='black')
            for f in faces:
                draw.rectangle((px+f['x']*im.width,py+f['y']*im.height,px+(f['x']+f['width'])*im.width,py+(f['y']+f['height'])*im.height),outline='red',width=2)
            m=mask(x.shape,faces);overlay=np.uint8(np.clip(x*.55+np.stack([m,np.zeros_like(m),np.zeros_like(m)],axis=-1)*.45,0,1)*255+.5);im2=Image.fromarray(overlay);im2.thumbnail((285,170));sheet.paste(im2,(307+(285-im2.width)//2,py))
        name=f'detections-{start//8+1:02d}.jpg';sheet.save(out/name,quality=90);pages.append(name)
    summary={}
    for strength in (0.,.3):
        sub=[r for r in rows if r['strength']==strength];g=np.array([r['grid_rmse'] for r in sub]);f=np.array([r['face_rmse'] for r in sub]);summary[str(strength)]={'heldout':len(sub),'face_lower_than_grid':int(np.sum(f<g)),'grid_mean_rmse':float(g.mean()),'face_mean_rmse':float(f.mean())}
    morph.dump({'scope':'Synthetic target and fitted model use the same detection-derived face mask, so favorable match is expected. Face detections not manually validated. Fixed scene, one album, one representative; no real face-edit or unseen-album validation. No semantic segmentation.','settings':settings,'summary':summary,'rows':rows,'detection_sheets':pages},out/'report.json')
if __name__=='__main__':main()
