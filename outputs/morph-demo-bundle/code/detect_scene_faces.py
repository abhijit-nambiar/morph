#!/usr/bin/env python3
"""Run a compiled local face detector and bind its results to source checksums."""
import argparse,json,subprocess
from pathlib import Path
import morph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups');p.add_argument('--detector',required=True,help='Compiled detect_faces.swift executable');p.add_argument('--out',required=True);a=p.parse_args()
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));paths=[]
    for g in groups:
        for photo in g['photos']:
            path=str(Path(photo).resolve())
            if path not in paths:paths.append(path)
    if not paths:raise ValueError('No photos')
    morph.protect_inputs([a.groups,a.detector,*paths],[a.out]);morph.protect_existing([a.out],False)
    before={path:morph.sha256_file(path) for path in paths}
    run=subprocess.run([a.detector,*paths],check=True,capture_output=True,text=True);rows=json.loads(run.stdout)
    if not isinstance(rows,list):raise ValueError('Detector did not return a list')
    by_path={}
    for row in rows:
        path=str(Path(row['path']).resolve())
        if path not in before or path in by_path:raise ValueError('Detector returned unexpected or duplicate image: '+path)
        by_path[path]=row
    if set(by_path)!=set(paths):raise ValueError('Detector omitted images')
    result=[]
    for path in paths:
        if morph.sha256_file(path)!=before[path]:raise ValueError('Image changed during detection: '+path)
        row=by_path[path];row['path']=path;row['source_sha256']=before[path];result.append(row)
    morph.dump(result,a.out)
    print(len(result),'images;',sum(bool(r.get('error')) for r in result),'detector errors;',sum(not r.get('faces') and not r.get('error') for r in result),'without faces')
if __name__=='__main__':main()
