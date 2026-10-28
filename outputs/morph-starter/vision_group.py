#!/usr/bin/env python3
"""Propose reviewable content-aware subgroups using a local macOS Vision helper."""
import argparse,html,json,math,os,subprocess,sys,tempfile
from pathlib import Path
import morph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('originals');p.add_argument('--extractor',required=True,help='Compiled vision_scene_features helper');p.add_argument('--out',required=True);p.add_argument('--gap-minutes',type=float,default=5);p.add_argument('--threshold',type=float,default=.56);a=p.parse_args()
    if not math.isfinite(a.gap_minutes) or a.gap_minutes<=0 or not math.isfinite(a.threshold) or not 0<a.threshold<2:raise ValueError('Invalid gap or threshold')
    folder=Path(a.originals).resolve();extractor=Path(a.extractor).resolve();out=Path(a.out).resolve()
    if not folder.is_dir() or not extractor.is_file():raise ValueError('Originals directory or extractor missing')
    if out.exists():raise ValueError('Output already exists')
    photos=morph.files(folder)
    if not photos:raise ValueError('No supported photos found')
    if len(photos)>1000:raise ValueError('Experimental distance matrix is limited to 1000 photos')
    if not out.parent.is_dir():raise ValueError('Output parent directory does not exist')
    before={str(path):morph.sha256_file(path) for path in photos}
    with tempfile.TemporaryDirectory(prefix='.vision-group-',dir=out.parent) as td:
        stage=Path(td);pathlist=stage/'paths.json';morph.dump(list(before),pathlist)
        groups=morph.chronological_groups(photos,a.gap_minutes);morph.dump({'version':1,'method':'chronological baseline','groups':groups},stage/'chronological.json')
        completed=subprocess.run([str(extractor),str(pathlist)],check=True,capture_output=True)
        distances=json.loads(completed.stdout);morph.dump(distances,stage/'distances.json')
        if any(morph.sha256_file(path)!=digest for path,digest in before.items()):raise ValueError('Input changed during feature extraction')
        script=Path(__file__).with_name('vision_cluster.py');subprocess.run([sys.executable,str(script),str(stage/'chronological.json'),str(stage/'distances.json'),'--threshold',str(a.threshold),'--out',str(stage/'groups.json')],check=True)
        group_data=json.loads((stage/'groups.json').read_text());group_data['source_groups']='chronological.json';group_data['source_manifest_sha256']=before;morph.dump(group_data,stage/'groups.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('review_groups.py')),str(stage/'groups.json'),'--source',str(folder),'--out',str(stage/'review')],check=True)
        if any(morph.sha256_file(path)!=digest for path,digest in before.items()):raise ValueError('Input changed during group review')
        index=json.loads((stage/'review'/'index.json').read_text());source={g['id']:g['source_scene'] for g in group_data['groups']}
        parts=['<!doctype html><meta charset="utf-8"><title>Scene grouping review</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;background:#f5f5f3;color:#222}article{background:white;padding:16px;margin:20px 0;border-radius:8px}img{max-width:100%;height:auto}</style><h1>Content-aware scene proposals</h1><p>Contact sheets are thumbnails. Review each proposal before editing; the grouping threshold is experimental.</p>']
        for g in index['groups']:
            sid=html.escape(g['id']);src=html.escape(source[g['id']]);img=html.escape(g['contact_sheet'],quote=True);parts.append(f'<article><h2>{sid} · {g["photo_count"]} photos</h2><p>From chronological {src}</p><img src="{img}" alt="Photos in {sid}"></article>')
        (stage/'review'/'index.html').write_text(''.join(parts),encoding='utf-8')
        pathlist.unlink();os.rename(stage,out)
    print('Review:',out/'review'/'index.html',flush=True)
if __name__=='__main__':main()
