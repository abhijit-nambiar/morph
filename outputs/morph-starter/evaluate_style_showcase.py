#!/usr/bin/env python3
"""Evaluate varied-album restoration and styled JPEG exports; evaluation inputs stay here."""
import argparse
import json
from pathlib import Path

import numpy as np

import cubic_color
import morph
import photometric_normalization as normalization


def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))


def summary(rows,before,after):
    return {'photos':len(rows),'mean_before':float(np.mean([r[before] for r in rows])),
            'mean_after':float(np.mean([r[after] for r in rows])),
            'p95_after':float(np.percentile([r[after] for r in rows],95)),
            'improved':sum(r[after]<r[before] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--size',type=int,default=1000)
    parser.add_argument('--out',required=True);args=parser.parse_args()
    root=Path(args.root).resolve()
    groups=json.loads((root/'groups.json').read_text())['groups']
    styles=json.loads((root/'styles.json').read_text())
    truth=json.loads((root/'varied-album/evaluation-truth.json').read_text())
    estimates=json.loads((root/'normalization-final.json').read_text())
    baseline=json.loads((root/'clean-baseline-hybrid.json').read_text())
    calibrated={g['representative'] for g in groups}
    models={(style,g['id']):cubic_color.deserialize(json.loads(
        (root/'models'/style/'fits'/g['id']/'fit.json').read_text())['fitted_recipe'])
        for style in styles for g in groups}
    rows=[]
    for item in truth['photos']:
        source,varied=item['source'],item['varied']
        clean=morph.load(source,args.size);image=morph.load(varied,args.size)
        if clean.shape!=image.shape:raise ValueError('Evaluation image dimensions differ')
        intended=normalization.render(clean,baseline['corrections'][source])
        correction=estimates['corrections'][varied]
        fixed=normalization.render(image,correction)
        row={'photo':Path(source).name,'source':source,'varied':varied,'scene':item['scene'],
             'calibration_counterpart':source in calibrated,'status':correction['status'],
             'normalization_before':rmse(image,intended),'normalization_after':rmse(fixed,intended),
             'bias_before':np.mean(image-intended,axis=(0,1)).tolist(),
             'bias_after':np.mean(fixed-intended,axis=(0,1)).tolist(),'styles':{}}
        # The known injected settings are consulted only by this evaluator.
        settings=item['settings'];base=baseline['corrections'][source]
        expected_scale=base['scale']/(1+settings['contrast'])
        expected_offset=np.asarray(base['offset'])-expected_scale*(
            settings['exposure_stops']*np.log(2)+np.asarray(settings['white_balance']))
        row['expected_correction']={'scale':float(expected_scale),'offset':expected_offset.tolist()}
        row['estimated_correction']={'scale':correction['scale'],'offset':correction['offset']}
        for style,spec in styles.items():
            target=morph.pillow_soft_target(intended,'neutral',spec['settings'])
            without=cubic_color.render(image,models[style,item['scene']])
            export=root/'final-albums'/style/item['scene']/'exports'/Path(varied).name
            actual=morph.load(export,args.size)
            if actual.shape!=target.shape:raise ValueError('Export geometry differs from evaluation target')
            row['styles'][style]={'before':rmse(without,target),'after':rmse(actual,target),
                                  'export':str(export)}
        rows.append(row)
        print(item['scene'],Path(source).name,'evaluated',flush=True)
    norm=summary(rows,'normalization_before','normalization_after')
    norm['estimated']=sum(r['status']=='estimated' for r in rows);norm['review']=norm['photos']-norm['estimated']
    before=[];after=[];scene_rows=[]
    for group in groups:
        subset=[r for r in rows if r['scene']==group['id']]
        b=np.mean([r['bias_before'] for r in subset],axis=0);a=np.mean([r['bias_after'] for r in subset],axis=0)
        before.append(b);after.append(a)
        scene_rows.append({'scene':group['id'],'photos':len(subset),'bias_before':b.tolist(),
                           'bias_after':a.tolist(),**summary(subset,'normalization_before','normalization_after')})
    norm['scene_bias_before']=float(np.sqrt(np.mean(np.asarray(before)**2)))
    norm['scene_bias_after']=float(np.sqrt(np.mean(np.asarray(after)**2)))
    norm['album_bias_before']=float(np.sqrt(np.mean(np.var(before,axis=0))))
    norm['album_bias_after']=float(np.sqrt(np.mean(np.var(after,axis=0))))
    norm['noncalibration']=summary([r for r in rows if not r['calibration_counterpart']],
                                  'normalization_before','normalization_after')
    style_summary={}
    for style in styles:
        values=[r['styles'][style] for r in rows]
        noncalibration=[r['styles'][style] for r in rows if not r['calibration_counterpart']]
        style_summary[style]={**summary(values,'before','after'),
                              'noncalibration':summary(noncalibration,'before','after')}
    morph.protect_existing([args.out],False)
    morph.dump({'scope':'Controlled same-album research evaluation. The correction stage sees the varied '
                'album and one intended reference per lighting group. Clean counterparts and injected settings '
                'are reserved for this evaluator. The intended normalization baseline is the same matcher run '
                'on the unperturbed album, so these scores establish stability against injected global tone/color '
                'variation, not an independent judgment of ideal normalization. Targets then use the independent '
                'Pillow style editor. Style weights are fitted only on representatives. Metrics cover decoded '
                'full-resolution JPEG exports sampled at the stated evaluation size. Original scenes, styles and '
                'algorithm were developed on this album; this is not unseen-album or human-edit validation. '
                'Local relighting, clipped-detail recovery and mask quality are outside this test.',
                'size':args.size,'variance_seed':truth['seed'],'calibration_photos':len(calibrated),
                'normalization':norm,'styles':style_summary,'scenes':scene_rows,'photos':rows},args.out)
    print(json.dumps({'normalization':norm,'styles':style_summary},indent=2),flush=True)


if __name__=='__main__':main()
