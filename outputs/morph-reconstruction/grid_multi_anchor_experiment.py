#!/usr/bin/env python3
"""Compare one- and multi-anchor grid fits on fixed held-out real and synthetic pairs."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,grid_light
import offcenter_light_experiment as synthetic


def fit_joint(pairs,penalty):
    normal=np.zeros((37,37));rhs=np.zeros((37,3))
    for x,y in pairs:
        rng=np.random.default_rng(0);idx=rng.choice(x.shape[0]*x.shape[1],min(x.shape[0]*x.shape[1],30000),replace=False)
        a=grid_light.features(x).reshape(-1,37)[idx].astype(np.float64);b=(y-x).reshape(-1,3)[idx].astype(np.float64)
        normal+=a.T@a/len(a);rhs+=a.T@b/len(a)
    return np.linalg.lstsq(normal/len(pairs)+penalty*np.eye(37),rhs/len(pairs),rcond=None)[0]


def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('split_report');p.add_argument('edits');p.add_argument('settings');p.add_argument('--out',required=True);p.add_argument('--size',type=int,default=256);p.add_argument('--regularization',type=float,default=.001);a=p.parse_args()
    if not np.isfinite(a.regularization) or a.regularization<0:raise ValueError('Invalid regularization')
    report=json.loads(Path(a.split_report).read_text());settings=morph.soft_settings('neutral',json.loads(Path(a.settings).read_text()));rows=[];scenes=[]
    for scene in report['cases']:
        anchors=scene['anchors'];held=[r['photo'] for r in scene['heldout']]
        if len(anchors)<2 or not held:continue
        assert scene['representative']==anchors[0]
        originals={p:morph.load(p,a.size) for p in anchors+held}
        for target_name in ('real','offcenter'):
            def target(path):
                x=originals[path]
                return morph.load(Path(a.edits)/Path(path).name,a.size) if target_name=='real' else synthetic.target(x,settings,(.2,.3),.3)
            pairs=[(originals[path],target(path)) for path in anchors]
            single=fit_joint(pairs[:1],a.regularization);joint=fit_joint(pairs,a.regularization)
            outcomes=[]
            for path in held:
                x=originals[path];y=target(path);one=grid_light.render(x,single);multi=grid_light.render(x,joint)
                row={'scene':scene['scene'],'photo':path,'target':target_name,'single_rmse':rmse(one,y),'joint_rmse':rmse(multi,y),'source_rmse':rmse(x,y),'single_out_of_gamut_fraction':grid_light.out_of_gamut(x,single),'joint_out_of_gamut_fraction':grid_light.out_of_gamut(x,joint)};rows.append(row);outcomes.append(row)
            scenes.append({'scene':scene['scene'],'target':target_name,'representative':anchors[0],'anchors':anchors,'single_mapping':grid_light.serialize(single),'joint_mapping':grid_light.serialize(joint),'heldout':outcomes})
        print(scene['scene'],len(anchors),'anchors',len(held),'held-out',flush=True)
    if not rows:raise ValueError('No multi-anchor scenes with held-out photos')
    summary={}
    for target in ('real','offcenter'):
        subset=[r for r in rows if r['target']==target];x=np.array([r['single_rmse'] for r in subset]);y=np.array([r['joint_rmse'] for r in subset]);summary[target]={'heldout':len(subset),'joint_lower':int(np.sum(y<x)),'single_mean_rmse':float(x.mean()),'joint_mean_rmse':float(y.mean()),'single_median_rmse':float(np.median(x)),'joint_median_rmse':float(np.median(y)),'single_mean_out_of_gamut_fraction':float(np.mean([r['single_out_of_gamut_fraction'] for r in subset])),'joint_mean_out_of_gamut_fraction':float(np.mean([r['joint_out_of_gamut_fraction'] for r in subset]))}
    morph.dump({'scope':'One album, same held-out photos and previously selected anchor splits, equal per-image fit weighting, same deterministic pixel samples and regularization. Real same-size pairs may be misaligned or edited differently; synthetic uses a fixed frame-relative target in all scene photos and a related smooth basis. No unseen album, subjective or professional-quality validation.', 'size':a.size,'regularization':a.regularization,'summary':summary,'scenes':scenes},a.out)
if __name__=='__main__':main()
