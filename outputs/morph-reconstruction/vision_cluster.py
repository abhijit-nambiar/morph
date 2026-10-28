#!/usr/bin/env python3
"""Propose scene subgroups from Vision feature distances within chronological groups."""
import argparse,json,math
from pathlib import Path
import numpy as np
import morph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('chronological_groups');p.add_argument('distances');p.add_argument('--threshold',type=float,default=.56);p.add_argument('--out',required=True);a=p.parse_args()
    if not math.isfinite(a.threshold) or not 0<a.threshold<2:raise ValueError('Invalid distance threshold')
    groups=morph.validate_groups(json.loads(Path(a.chronological_groups).read_text()));data=json.loads(Path(a.distances).read_text());paths=data['paths'];m=np.asarray(data['distances'],dtype=float)
    if len(set(paths))!=len(paths) or m.shape!=(len(paths),len(paths)) or not np.isfinite(m).all() or np.any(m<0) or not np.allclose(m,m.T,atol=1e-5) or not np.allclose(np.diag(m),0,atol=1e-5):raise ValueError('Invalid distances')
    lookup={str(Path(path).resolve()):i for i,path in enumerate(paths)};expected={str(Path(path).resolve()) for g in groups for path in g['photos']}
    if len(lookup)!=len(paths) or set(lookup)!=expected:raise ValueError('Distance paths must exactly match grouped photos')
    result=[];splits=[]
    for g in groups:
        members=[str(Path(x).resolve()) for x in g['photos']];clusters=[[x] for x in members]
        while len(clusters)>1:
            choices=[]
            for i in range(len(clusters)):
                for j in range(i):
                    dist=float(np.mean([m[lookup[x],lookup[y]] for x in clusters[i] for y in clusters[j]]));choices.append((dist,j,i))
            dist,j,i=min(choices)
            if dist>a.threshold:break
            clusters[j].extend(clusters.pop(i))
        splits.append({'original_scene':g['id'],'photos':len(members),'proposed_subgroups':len(clusters)})
        for cluster in clusters:
            medoid=min(cluster,key=lambda x:(sum(m[lookup[x],lookup[y]] for y in cluster),x))
            result.append({'id':f'scene-{len(result)+1:02d}','representative':medoid,'photos':cluster,'source_scene':g['id']})
    morph.protect_inputs([a.chronological_groups,a.distances]+list(expected),[a.out]);morph.protect_existing([a.out],False)
    morph.dump({'version':1,'method':'experimental Vision feature distances, average linkage inside chronological groups; manually review','threshold':a.threshold,'vision_revision':data.get('revision'),'scope':'Image feature distances are not semantic scene labels or segmentation. This proposal never merges across chronological groups, and the threshold is not validated. Verify groups and source files before fitting.','source_groups':str(Path(a.chronological_groups).resolve()),'splits':splits,'groups':result},a.out)
    print(len(expected),'photos;',len(result),'proposed groups')
if __name__=='__main__':main()
