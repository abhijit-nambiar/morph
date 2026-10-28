#!/usr/bin/env python3
"""Test a fixed second prompt-generated reference against one-reference scene fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color,prompt_target

PROMPTS=['warm and slightly soft contrast and lift shadows','cool and strong contrast and deepen shadows','warmer and reduce highlights and muted blues']

def rmse(x,y):return float(np.sqrt(np.mean((x-y)**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=128);a=p.parse_args()
    if not 64<=a.size<=512:raise ValueError('Invalid size')
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));inputs=[a.groups]+[photo for g in groups for photo in g['photos']];morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False);runs=[]
    for prompt in PROMPTS:
        settings,interpretation=prompt_target.parse(prompt);rows=[];scenes=[]
        for g in groups:
            rep=g['representative'];alternatives=[x for x in g['photos'] if x!=rep]
            if len(alternatives)<2:continue
            second=alternatives[0];heldout=alternatives[1:];weights=[]
            for path in (rep,second):
                x=morph.load(path,a.size);target=morph.pillow_soft_target(x,'neutral',settings);w,_=poly_color.fit(x,target,regularization=.001);weights.append(w)
            pooled=np.mean(weights,axis=0)
            for path in heldout:
                x=morph.load(path,a.size);target=morph.pillow_soft_target(x,'neutral',settings)
                rows.append({'scene':g['id'],'photo':path,'source_rmse':rmse(x,target),'one_reference_rmse':rmse(poly_color.render(x,weights[0]),target),'two_reference_rmse':rmse(poly_color.render(x,pooled),target)})
            scenes.append({'scene':g['id'],'references':[rep,second],'heldout':len(heldout)})
        if not rows:raise ValueError('No held-out photos')
        one=np.array([r['one_reference_rmse'] for r in rows]);two=np.array([r['two_reference_rmse'] for r in rows]);source=np.array([r['source_rmse'] for r in rows]);runs.append({'prompt':prompt,'interpretation':interpretation,'summary':{'heldout':len(rows),'source_mean_rmse':float(source.mean()),'one_reference_mean_rmse':float(one.mean()),'two_reference_mean_rmse':float(two.mean()),'two_lower':int(np.sum(two<one-1e-8)),'two_higher':int(np.sum(two>one+1e-8))},'scenes':scenes,'photos':rows});print(prompt,len(rows),'held-out',flush=True)
    morph.dump({'scope':'Controlled same-album experiment: both reference targets and independent held-out targets come from the same deterministic global prompt renderer. The first reference is the existing group representative, and the second is the first different photo in the recorded group order. Both references are excluded from evaluation. Fit separately and average polynomial coefficients. Groups are not verified, and generated targets are not human edits. Reduced-resolution RGB RMSE is not visual or professional quality. This test does not evaluate face-relative or frame-relative edits or automatic normalization.','size':a.size,'regularization':.001,'runs':runs},a.out)
if __name__=='__main__':main()
