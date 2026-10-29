#!/usr/bin/env python3
"""Reproduce a styled, varied, normalized, verified offline album showcase."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import morph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('originals');parser.add_argument('--scenes',required=True)
    parser.add_argument('--styles',required=True);parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--size',type=int,default=512);parser.add_argument('--evaluation-size',type=int,default=1000)
    parser.add_argument('--jobs',type=int,default=1);parser.add_argument('--out',required=True)
    parser.add_argument('--person-experiment',help='Optional historical person run for a separate labeled local-light example')
    args=parser.parse_args();out=Path(args.out).resolve();originals=Path(args.originals).resolve()
    if out.exists():raise ValueError('Use a new output folder')
    plan=json.loads(Path(args.scenes).read_text());groups=[]
    for group in plan['groups']:
        names=[Path(p).name for p in group['photos']]
        photos=[str(originals/name) for name in names]
        groups.append({**group,'photos':photos,
                       'representative':str(originals/Path(group['representative']).name)})
    morph.validate_groups({'groups':groups})
    if {p for g in groups for p in g['photos']}!={str(p) for p in morph.files(originals)}:
        raise ValueError('Scene plan must cover the originals folder exactly')
    styles=json.loads(Path(args.styles).read_text())
    if any(len(g['photos'])<1 for g in groups):raise ValueError('Empty scene')
    out.mkdir(parents=True);logs=out/'logs';logs.mkdir()
    morph.dump({'version':1,'method':plan.get('method','Reviewed scene plan'),'groups':groups},out/'groups.json')
    morph.dump(styles,out/'styles.json');morph.dump({g['id']:g['representative'] for g in groups},out/'references.json')
    state={'status':'running','completed_stages':[],'seed':args.seed}
    code=Path(__file__).parent
    def run(stage,script,*arguments):
        state['active_stage']=stage;morph.dump(state,out/'run.json');print(stage,flush=True)
        with (logs/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(code/script),*map(str,arguments)],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed_stages'].append(stage);state.pop('active_stage',None);morph.dump(state,out/'run.json')
    try:
        run('benchmark','benchmark_styles.py',out/'groups.json',out/'styles.json','--size',args.size,'--out',out/'benchmark-final.json')
        benchmark=json.loads((out/'benchmark-final.json').read_text())
        # Cubic support is kept explicit in the evaluator and package. Selection
        # is recorded; a different winner needs review before using this showcase.
        if benchmark['selected_spec']['renderer']!='cubic':raise ValueError('Validation selected a different renderer; inspect benchmark')
        run('variances','make_variances.py',out/'groups.json','--seed',args.seed,'--out',out/'varied-album')
        run('normalization','photometric_normalization.py',out/'varied-album/groups.json',
            '--references',out/'varied-album/references.json','--out',out/'normalization-final.json',
            '--corrections-out',out/'corrections-final.json')
        run('evaluation-baseline','photometric_normalization.py',out/'groups.json',
            '--references',out/'references.json','--out',out/'clean-baseline-hybrid.json')
        run('fit','fit_style_album.py',out/'groups.json',out/'styles.json','--size',args.size,
            '--regularization',benchmark['selected_spec']['regularization'],'--out',out/'models')
        run('replay','apply_normalized_styles.py',out/'varied-album/groups.json',out/'styles.json',out/'models',
            out/'normalization-final.json',out/'corrections-final.json','--jobs',args.jobs,'--out',out/'final-albums')
        run('evaluate','evaluate_style_showcase.py',out,'--size',args.evaluation_size,'--out',out/'evaluation.json')
        run('verify-exports','verify_style_showcase.py',out,'--out',out/'verification.json')
        showcase_arguments = ['--person-experiment', str(Path(args.person_experiment).resolve())] if args.person_experiment else []
        run('showcase','build_style_showcase.py',out,*showcase_arguments)
        run('verify-artifact','verify_showcase_artifact.py',out,'--out',out/'artifact-verification.json')
        state['status']='complete';morph.dump(state,out/'run.json');print(out/'index.html',flush=True)
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);morph.dump(state,out/'run.json');raise


if __name__=='__main__':main()
