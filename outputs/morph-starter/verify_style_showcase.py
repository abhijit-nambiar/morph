#!/usr/bin/env python3
"""Verify all replay inputs and full-resolution styled exports independently of scores."""
import argparse
import json
from pathlib import Path

from PIL import Image

import morph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--out',required=True);args=parser.parse_args()
    root=Path(args.root).resolve();styles=json.loads((root/'styles.json').read_text())
    groups=morph.validate_groups(json.loads((root/'final-albums/replay-groups.json').read_text()))
    normalization=json.loads((root/'normalization-final.json').read_text());issues=[];rows=[]
    for source,digest in normalization['source_manifest'].items():
        if morph.sha256_file(source)!=digest:issues.append('Changed normalization input: '+source)
    for style in styles:
        report=json.loads((root/'final-albums'/style/'review.json').read_text())
        recorded={(r['scene'],str(Path(r['original']).resolve())):Path(r['export']).resolve()
                  for r in report['photos']}
        if len(recorded)!=len(report['photos']):issues.append('Duplicate export record: '+style)
        expected=set()
        for group in groups:
            # Original reference checksums remain bound to the trained fit.
            fit=json.loads((root/'models'/style/'fits'/group['id']/'fit.json').read_text())
            for entry in fit['source_manifest']:
                if morph.sha256_file(entry['path'])!=entry['sha256']:
                    issues.append('Changed fitted source: '+entry['path'])
            for source in group['photos']:
                key=(group['id'],str(Path(source).resolve()));expected.add(key)
                dest=root/'final-albums'/style/group['id']/'exports'/Path(source).name
                problems=[];size=None
                if recorded.get(key)!=dest:problems.append('missing or unexpected export record')
                try:
                    with Image.open(source) as original,Image.open(dest) as result:
                        size=morph.oriented_size(original)
                        if result.size!=size:problems.append('oriented dimensions differ')
                        if not result.info.get('icc_profile'):problems.append('missing ICC profile')
                        if original.getexif() and not result.getexif():problems.append('missing EXIF')
                        result.load()
                except Exception as exc:problems.append(str(exc))
                rows.append({'style':style,'scene':group['id'],'source':source,'export':str(dest),
                             'dimensions':size,'problems':problems})
                if problems:issues.append(style+'/'+Path(source).name+': '+', '.join(problems))
        if set(recorded)!=expected:issues.append('Unexpected export records: '+style)
        expected_files={Path(r['export']).resolve() for r in rows if r['style']==style}
        actual={p.resolve() for p in (root/'final-albums'/style).glob('*/exports/*') if p.is_file()}
        if actual!=expected_files:issues.append('Missing or unexpected export files: '+style)
    morph.protect_existing([args.out],False)
    morph.dump({'scope':'Input checksums, recorded coverage, complete image decoding, original oriented '
                'dimensions, ICC and EXIF presence checked. No perceptual-quality claim.',
                'exports':len(rows),'album_exports':sum(Path(r['source']).name.startswith('varied-') for r in rows),
                'problems':len(issues),'issues':issues,'photos':rows},args.out)
    print(len(rows),'full-resolution exports verified;',len(issues),'problems',flush=True)
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
