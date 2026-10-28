#!/usr/bin/env python3
"""Exploratory fitting on unverified estimated crops; no original export."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import morph

def main():
    p=argparse.ArgumentParser();p.add_argument('crop_report');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);a=p.parse_args()
    data=json.loads(Path(a.crop_report).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];visuals=[]
    for rec in data['rows']:
        x=morph.load(rec['original'],1200); target=morph.load(rec['target'],a.size);h,w=x.shape[:2];b=rec['estimate']['box'];left=int(b[0]*w);top=int(b[1]*h);right=int(b[2]*w);bottom=int(b[3]*h)
        if not (0<=left<right<=w and 0<=top<bottom<=h):raise ValueError('Invalid crop '+rec['file'])
        crop=Image.fromarray(np.uint8(x[top:bottom,left:right]*255+.5)).resize((target.shape[1],target.shape[0]),Image.Resampling.LANCZOS);original=np.asarray(crop,dtype=np.float32)/255
        fit,metrics=morph.fit(original,target);pred=morph.render(original,fit);rows.append({'file':rec['file'],'crop_estimate':rec['estimate'],'recipe':morph.recipe(fit),'metrics':metrics});visuals.append((rec['file'],(original,target,pred)));print(rec['file'],round(metrics['rmse'],4),flush=True)
    pages=morph.comparison_sheet(visuals,out,labels='estimated crop / human edit / reconstruction');morph.dump({'warning':'Exploratory results use automatically estimated, unreviewed crops. Resampling and imperfect alignment affect metrics; target crops are never used in exports.','results':rows,'comparison_sheets':pages},out/'report.json')
if __name__=='__main__':main()
