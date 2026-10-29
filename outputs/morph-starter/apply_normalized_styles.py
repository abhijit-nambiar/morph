#!/usr/bin/env python3
"""Replay saved style fits with estimated corrections using the existing apply path."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import morph
import photometric_normalization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups');parser.add_argument('styles');parser.add_argument('models')
    parser.add_argument('normalization');parser.add_argument('corrections');parser.add_argument('--out',required=True)
    parser.add_argument('--jobs',type=int,default=1)
    parser.add_argument('--resume',action='store_true',help='Regenerate planned outputs of this same run')
    args=parser.parse_args()
    groups=morph.validate_groups(json.loads(Path(args.groups).read_text()))
    styles=json.loads(Path(args.styles).read_text())
    normalization=json.loads(Path(args.normalization).read_text())
    config=json.loads(Path(args.corrections).read_text())
    expected={g['id']:{Path(p).name:photometric_normalization.gamut_recipe(normalization['corrections'][p])
                       for p in g['photos']} for g in groups}
    if config!={'photos':expected}:raise ValueError('Correction file differs from normalization report')
    for path,digest in normalization['source_manifest'].items():
        if morph.sha256_file(path)!=digest:raise ValueError('Normalization input changed: '+path)
    out=Path(args.out).resolve()
    if not 1<=args.jobs<=3:raise ValueError('jobs must be 1–3')
    if out.exists() and not args.resume:raise ValueError('Use a new output folder or --resume')
    out.mkdir(parents=True,exist_ok=True)
    references={s['scene']:s['reference'] for s in normalization['scenes']}
    replay_groups=[]
    for group in groups:
        reference=references[group['id']]
        replay_groups.append({**group,'representative':reference,
                              'photos':[reference]+group['photos']})
    morph.validate_groups({'groups':replay_groups})
    replay_file=out/'replay-groups.json'
    if args.resume and replay_file.exists() and json.loads(replay_file.read_text())!={'groups':replay_groups}:
        raise ValueError('Existing replay groups differ; use a new output folder')
    morph.dump({'groups':replay_groups},replay_file)
    state={'status':'running','completed_styles':[],
           'scope':'New copies of the same album intentionally replay clean-reference style fits. '
                   'The existing explicit input-change option permits these copies; their current '
                   'checksums are verified against the normalization manifest before rendering.'}
    morph.dump(state,out/'run.json')
    try:
        def replay(style):
            command=[sys.executable,str(Path(morph.__file__)),'apply',str(replay_file),
                            str(Path(args.models)/style/'fits'),'--allow-changed-inputs',
                            '--corrections',args.corrections,'--out',str(out/style)]
            if args.resume:command.append('--overwrite')
            subprocess.run(command,check=True)
            return style
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures=[executor.submit(replay,style) for style in styles]
            for future in as_completed(futures):
                state['completed_styles'].append(future.result());morph.dump(state,out/'run.json')
        state['status']='complete';morph.dump(state,out/'run.json')
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);morph.dump(state,out/'run.json');raise


if __name__=='__main__':main()
