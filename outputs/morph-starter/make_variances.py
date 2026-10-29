#!/usr/bin/env python3
"""Create controlled scene/photo lighting variations; keep truth separate from inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

import morph


def vary(x, settings):
    v=np.clip(x,1e-5,1-1e-5)
    z=(1+settings['contrast'])*np.log(v/(1-v))
    z+=settings['exposure_stops']*np.log(2)+np.asarray(settings['white_balance'],dtype=np.float32)
    return 1/(1+np.exp(-z))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups');parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--size',type=int,default=0,help='0 keeps original dimensions')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    groups=morph.validate_groups(json.loads(Path(args.groups).read_text()))
    out=Path(args.out).resolve()
    if out.exists() or args.size<0:
        raise ValueError('Use a new output folder and nonnegative size')
    rng=np.random.default_rng(args.seed);out.mkdir(parents=True)
    varied_groups=[];truth=[];references={}
    for group in groups:
        scene_ev=float(rng.uniform(-.22,.22));scene_wb=float(rng.uniform(-.055,.055))
        scene_contrast=float(rng.uniform(-.035,.035));photos=[]
        references[group['id']]=group['representative']
        for source in group['photos']:
            ev=scene_ev+float(rng.uniform(-.18,.18));wb=scene_wb+float(rng.uniform(-.035,.035))
            settings={'exposure_stops':ev,'contrast':scene_contrast+float(rng.uniform(-.025,.025)),
                      'white_balance':[wb,float(rng.uniform(-.02,.02)),-wb]}
            dest=out/'inputs'/('varied-'+Path(source).name)
            image=morph.load(source,args.size or None)
            morph.save_render(image,None,dest,renderer=lambda tile,unused:vary(tile,settings),source=source)
            photos.append(str(dest));truth.append({'source':source,'varied':str(dest),
                                                   'scene':group['id'],'settings':settings,
                                                   'scene_settings':{'exposure_stops':scene_ev,
                                                                     'warm_cool':scene_wb,
                                                                     'contrast':scene_contrast}})
        representative=str(out/'inputs'/('varied-'+Path(group['representative']).name))
        varied_groups.append({'id':group['id'],'label':group.get('label',group['id']),
                              'representative':representative,'photos':photos})
        print(group['id'],len(photos),'variants generated',flush=True)
    morph.dump({'version':1,'method':'Controlled varied copies; scene IDs inherited from reviewed clean groups',
                'groups':varied_groups},out/'groups.json')
    morph.dump(references,out/'references.json')
    morph.dump({'scope':'Evaluation-only perturbation truth. Never supplied to normalization or fitting. '
                'Bounded logit tone curves simulate global settings changes; they do not simulate '
                'new shadows, clipping, sensor noise or RAW camera processing.',
                'seed':args.seed,'size':args.size,'photos':truth},out/'evaluation-truth.json')


if __name__=='__main__':main()
