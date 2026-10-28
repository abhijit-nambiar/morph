#!/usr/bin/env python3
"""Fill missing face detections using a second checksum-verified detector pass."""
import argparse,json
from pathlib import Path
import morph,face_scene_pipeline


def read(path):
    rows=json.loads(Path(path).read_text());result={}
    if not isinstance(rows,list):raise ValueError('Expected detection list')
    for row in rows:
        name=str(Path(row['path']).resolve())
        if name in result:raise ValueError('Duplicate detection: '+name)
        if row.get('source_sha256')!=morph.sha256_file(name):raise ValueError('Missing or stale detection checksum: '+name)
        if row.get('error'):raise ValueError('Detection error: '+name)
        face_scene_pipeline.checked_faces(row.get('faces'),allow_empty=True);result[name]=row
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline');p.add_argument('additional');p.add_argument('--out',required=True);a=p.parse_args()
    baseline=read(a.baseline);additional=read(a.additional)
    if set(additional)-set(baseline):raise ValueError('Additional detections contain unknown photos')
    output=[];replaced=[]
    for path,row in baseline.items():
        extra=additional.get(path)
        if extra and not row['faces'] and extra['faces']:
            merged=dict(extra);merged['detection_origin']='additional pass';output.append(merged);replaced.append(path)
        else:output.append(row)
    morph.protect_inputs([a.baseline,a.additional,*baseline.keys()],[a.out]);morph.protect_existing([a.out],False)
    morph.dump(output,a.out);print('Filled',len(replaced),'previously empty detections; boxes require visual review')
if __name__=='__main__':main()
