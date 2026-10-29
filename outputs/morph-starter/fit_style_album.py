#!/usr/bin/env python3
"""Make independent styled references and fit them through the existing album CLI."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import morph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('groups');parser.add_argument('styles');parser.add_argument('--out',required=True)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--renderer',choices=['polynomial','cubic'],default='cubic')
    parser.add_argument('--regularization',type=float,default=1e-6)
    args=parser.parse_args()
    groups=morph.validate_groups(json.loads(Path(args.groups).read_text()))
    styles=json.loads(Path(args.styles).read_text());out=Path(args.out).resolve()
    if out.exists():raise ValueError('Use a new output folder')
    out.mkdir(parents=True)
    for style_id,style in styles.items():
        if not style_id.replace('-','').isalnum():raise ValueError('Invalid style ID')
        directory=out/style_id;targets=directory/'targets'
        for group in groups:
            source=group['representative'];image=morph.load(source)
            morph.save_render(image,None,targets/Path(source).name,
                              renderer=lambda tile,unused:morph.pillow_soft_target(tile,'neutral',style['settings']),
                              source=source)
        morph.dump({'scope':'Independent Pillow tone LUT and HSV composite, authored by the agent. '
                    'The fitter reads source/target image pairs and receives no target control settings.',
                    'style':style,'targets':[{'source':g['representative'],
                                            'target':str(targets/Path(g['representative']).name),
                                            'target_sha256':morph.sha256_file(targets/Path(g['representative']).name)}
                                           for g in groups]},directory/'targets.json')
        subprocess.run([sys.executable,str(Path(morph.__file__)),'album',args.groups,
                        '--targets',str(targets),'--renderer',args.renderer,
                        '--solver','least-squares','--regularization',str(args.regularization),
                        '--size',str(args.size),'--preview-only','--out',str(directory/'fits')],check=True)
        print(style_id,'references and fits saved',flush=True)


if __name__=='__main__':main()
