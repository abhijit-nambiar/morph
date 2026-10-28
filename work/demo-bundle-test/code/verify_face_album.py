#!/usr/bin/env python3
"""Verify file coverage, image dimensions and color profiles of a face-album export."""
import argparse,json
from pathlib import Path
from PIL import Image
import morph,face_scene_pipeline


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fits');p.add_argument('exports');p.add_argument('--out',required=True);a=p.parse_args()
    fits=Path(a.fits);exports=Path(a.exports);index=json.loads((fits/'index.json').read_text());scenes=index['scenes'];records=[];issues=[];inputs=[fits/'index.json',exports/'report.json']
    for scene in scenes:
        fitfile=fits/scene/'fit.json';fit,_,sources=face_scene_pipeline.validate_fit(fitfile);inputs.append(fitfile)
        if fit['scene']!=scene:raise ValueError('Scene mismatch')
        for path,boxes in sources:
            dest=exports/scene/'exports'/Path(path).name;inputs.extend([path,dest]);problems=[]
            try:
                with Image.open(path) as src:original_size=morph.oriented_size(src);has_exif=bool(src.getexif())
                with Image.open(dest) as im:
                    size=im.size;icc=bool(im.info.get('icc_profile'));has_output_exif=bool(im.getexif());im.load()
                if tuple(size)!=tuple(original_size):problems.append('dimension mismatch')
                if not icc:problems.append('missing ICC profile')
                if has_exif and not has_output_exif:problems.append('source EXIF not present')
            except Exception as e:
                size=None;problems.append('unreadable or missing export: '+str(e))
            row={'scene':scene,'source':path,'export':str(dest),'dimensions':size,'detected_faces':len(boxes),'problems':problems};records.append(row)
            if problems:issues.append(row)
    expected={Path(r['export']).resolve() for r in records};actual={p.resolve() for scene in scenes for p in (exports/scene/'exports').glob('*') if p.is_file()};extra=sorted(str(x) for x in actual-expected)
    morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False)
    morph.dump({'scope':'Checks scene membership, original file checksums, decodability, orientation-adjusted dimensions, presence of ICC and EXIF. Does not evaluate visual quality, accuracy of boxes, preservation of all metadata, or processing artifacts.','expected':len(records),'problem_count':len(issues),'unexpected_files':extra,'photos_without_detected_faces':sum(not r['detected_faces'] for r in records),'photos':records},a.out)
    print(len(records),'photos;',len(issues),'with file problems;',len(extra),'unexpected files;',sum(not r['detected_faces'] for r in records),'without detected faces')
    if issues or extra:raise SystemExit(1)
if __name__=='__main__':main()
