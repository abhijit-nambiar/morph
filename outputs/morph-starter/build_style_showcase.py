#!/usr/bin/env python3
"""Build an offline photographic research showcase from verified album outputs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import morph
import photometric_normalization as normalization
from showcase_pipeline import pipeline_data, attach_person_example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--size',type=int,default=1600)
    parser.add_argument('--person-experiment',help='Optional historical person run to illustrate separately; never applied to styled albums')
    args=parser.parse_args();root=Path(args.root).resolve();assets=root/'assets'
    if (root/'index.html').exists() or assets.exists():raise ValueError('Showcase already exists')
    evaluation=json.loads((root/'evaluation.json').read_text())
    verification=json.loads((root/'verification.json').read_text())
    if verification['problems']:raise ValueError('Export verification has unresolved problems')
    groups=morph.validate_groups(json.loads((root/'groups.json').read_text()))
    styles=json.loads((root/'styles.json').read_text())
    truth=json.loads((root/'varied-album/evaluation-truth.json').read_text())['photos']
    estimates=json.loads((root/'normalization-final.json').read_text())
    baseline=json.loads((root/'clean-baseline-hybrid.json').read_text())
    by_source={r['source']:r for r in truth}
    calibration={g['representative'] for g in groups};photos=[]
    assets.mkdir()
    def save(array,name):
        path=assets/(name+'.jpg');morph.save(array,path,quality=93)
        return str(path.relative_to(root))
    for group in groups:
        for source in group['photos']:
            item=by_source[source];varied=item['varied'];pid=Path(source).stem
            clean=morph.load(source,args.size);image=morph.load(varied,args.size)
            normalized=normalization.render(image,estimates['corrections'][varied])
            canonical=normalization.render(clean,baseline['corrections'][source])
            record={'id':pid,'name':Path(source).name,'scene':group['id'],
                    'calibration':source in calibration,
                    'review':estimates['corrections'][varied]['status']!='estimated',
                    'assets':{'source':save(clean,pid+'-source'),
                              'varied':save(image,pid+'-varied'),
                              'normalized':save(normalized,pid+'-normalized'),'styles':{}},'full':{}}
            for style,spec in styles.items():
                export=root/'final-albums'/style/group['id']/'exports'/Path(varied).name
                result=morph.load(export,args.size)
                target=morph.pillow_soft_target(canonical,'neutral',spec['settings'])
                record['assets']['styles'][style]={'result':save(result,pid+'-'+style+'-result'),
                                                  'target':save(target,pid+'-'+style+'-target')}
                record['full'][style]=str(export.relative_to(root))
            photos.append(record)
        print(group['id'],'showcase assets prepared',flush=True)
    scenes=[]
    for group in groups:
        ids=[Path(p).stem for p in group['photos']];rep=Path(group['representative']).stem
        others=[pid for pid in ids if pid!=rep]
        chosen=([others[0],rep,others[-1]] if len(others)>=2 else ids[:3])
        scenes.append({'id':group['id'],'label':group.get('label',group['id']),
                       'photos':ids,'reference':rep,'showcase':chosen})
    reference_group=next((g for g in groups if g['id']=='scene-02'),max(groups,key=lambda g:len(g['photos'])))
    source=reference_group['representative'];pid=Path(source).stem
    reference={'source':next(p['assets']['source'] for p in photos if p['id']==pid),'styles':{}}
    for style in styles:
        target=root/'models'/style/'targets'/Path(source).name
        result=root/'final-albums'/style/reference_group['id']/'exports'/Path(source).name
        reference['styles'][style]={'target':save(morph.load(target,args.size),'reference-'+style+'-target'),
                                    'result':save(morph.load(result,args.size),'reference-'+style+'-result')}
    lookup={p['name']:p['id'] for p in photos}
    # Curated for clear composition rather than selected by reconstruction error.
    data={'styles':[{'id':sid,**spec} for sid,spec in styles.items()],
          'photos':photos,'scenes':scenes,'reference':reference,
          'hero':lookup.get('DSC_0995-22.jpg',pid),'transferScene':reference_group['id'],
          'normalizationPhoto':lookup.get('DSC_1284-57.jpg',photos[-1]['id']),
          'evaluation':evaluation,'verification':verification,
          'preview_scope':'Source/result previews have maximum side 1600 pixels. '
                          'Result previews come from actual full-resolution JPEG exports. '
                          'Per-photo target previews use the independent style generator on the '
                          'reference-calibrated clean baseline at preview resolution.'}
    data['pipeline'] = pipeline_data(root, groups, styles, estimates, evaluation)
    if args.person_experiment:
        data['localEdits'] = attach_person_example(root, args.person_experiment)
    # Keep UI data portable and avoid embedding private absolute filesystem paths.
    data['evaluation']={k:evaluation[k] for k in ('scope','normalization','styles')}
    data['verification']={k:verification[k] for k in ('exports','album_exports','problems')}
    template=Path(__file__).with_name('showcase_template.html').read_text()
    markup=template.replace('__MORPH_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    (root/'index.html').write_text(markup,encoding='utf-8')
    morph.dump(data,root/'showcase-data.json')
    print('Showcase:',root/'index.html',flush=True)


if __name__=='__main__':main()
