#!/usr/bin/env python3
"""Verify a full global/grid album replay against groups and fitted source manifests."""
import argparse,json
from pathlib import Path
from PIL import Image
import morph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('fits');p.add_argument('album');p.add_argument('--out',required=True);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));album=Path(a.album);report=json.loads((album/'review.json').read_text());records={}
    for r in report['photos']:
        key=(r['scene'],str(Path(r['original']).resolve()))
        if key in records:raise ValueError('Duplicate export record')
        records[key]=r
    expected=set();rows=[];inputs=[a.groups,album/'review.json']
    for g in groups:
        fitfile=Path(a.fits)/g['id']/'fit.json';fit=json.loads(fitfile.read_text());inputs.append(fitfile);manifest={r['path']:r['sha256'] for r in fit['source_manifest']}
        if set(manifest)!={str(Path(p).resolve()) for p in g['photos']}:raise ValueError('Fit membership differs from groups')
        for photo in g['photos']:
            path=str(Path(photo).resolve());key=(g['id'],path);expected.add(key);problems=[];record=records.get(key);inputs.append(path)
            if morph.sha256_file(path)!=manifest[path]:problems.append('source checksum changed')
            if not record:problems.append('missing export record');dest=None
            else:
                dest=Path(record['export']);inputs.append(dest)
                if dest.resolve()!=(album/g['id']/'exports'/Path(path).name).resolve():problems.append('unexpected export path')
                try:
                    with Image.open(path) as src:size=morph.oriented_size(src);exif=bool(src.getexif())
                    with Image.open(dest) as im:
                        if tuple(im.size)!=tuple(size):problems.append('dimension mismatch')
                        if not im.info.get('icc_profile'):problems.append('missing ICC profile')
                        if exif and not im.getexif():problems.append('missing EXIF')
                        im.load()
                except Exception as e:problems.append('unreadable export: '+str(e))
            rows.append({'scene':g['id'],'source':path,'export':str(dest) if dest else None,'problems':problems})
    extra=sorted(str(x) for x in set(records)-expected);expected_paths={(album/g['id']/'exports'/Path(p).name).resolve() for g in groups for p in g['photos']};actual={p.resolve() for g in groups for p in (album/g['id']/'exports').glob('*') if p.is_file()};extra_files=sorted(str(p) for p in actual-expected_paths);morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False)
    problems=sum(bool(r['problems']) for r in rows);morph.dump({'scope':'Checks source checksums, recorded coverage, decoding, dimensions, ICC and EXIF presence; does not judge visual quality or preserve all metadata.','photos':len(rows),'problem_count':problems,'extra_records':extra,'extra_files':extra_files,'rows':rows},a.out)
    print(len(rows),'photos;',problems,'with file problems;',len(extra),'unexpected records;',len(extra_files),'unexpected files')
    if problems or extra or extra_files:raise SystemExit(1)
if __name__=='__main__':main()
