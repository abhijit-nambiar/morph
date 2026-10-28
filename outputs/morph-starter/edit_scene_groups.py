#!/usr/bin/env python3
"""Export editable scene assignments and compile a checked replacement group file."""
import argparse,csv,json
from pathlib import Path
import morph

FIELDS=['source','scene','representative']


def build(a):
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));outputs=[a.out]
    morph.protect_inputs([a.groups]+[p for g in groups for p in g['photos']],outputs);morph.protect_existing(outputs,False)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for g in groups:
            for path in g['photos']:writer.writerow({'source':str(Path(path).resolve()),'scene':g['id'],'representative':'yes' if path==g['representative'] else ''})
    print(sum(len(g['photos']) for g in groups),'assignments written')


def compile_(a):
    groups=morph.validate_groups(json.loads(Path(a.groups).read_text()));original={str(Path(p).resolve()) for g in groups for p in g['photos']};inputs=[a.groups,a.assignments,*original]
    with open(a.assignments,newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS:raise ValueError('Unexpected assignment CSV columns')
        rows=list(reader)
    seen=set();scenes={}
    for row in rows:
        source=row['source'].strip();scene=row['scene'].strip();representative=row['representative'].strip().lower()
        if not source or not scene:raise ValueError('Every row needs a source and scene')
        path=str(Path(source).resolve())
        if path not in original:raise ValueError('Unknown source photo: '+source)
        if path in seen:raise ValueError('Duplicate source photo: '+source)
        if representative not in ('','yes'):raise ValueError('Representative must be yes or blank')
        seen.add(path);group=scenes.setdefault(scene,{'id':scene,'photos':[],'representative':None});group['photos'].append(path)
        if representative=='yes':
            if group['representative'] is not None:raise ValueError('Multiple representatives in scene '+scene)
            group['representative']=path
    if seen!=original:raise ValueError('Assignment file must include every original photo; missing '+str(len(original-seen)))
    missing=[scene for scene,g in scenes.items() if g['representative'] is None]
    if missing:raise ValueError('Exactly one representative is required per scene: '+', '.join(missing))
    result={'version':1,'method':'manually reviewed assignments','source_groups':str(Path(a.groups).resolve()),'source_groups_sha256':morph.sha256_file(a.groups),'groups':list(scenes.values())}
    morph.validate_groups(result);morph.protect_inputs(inputs,[a.out]);morph.protect_existing([a.out],False);morph.dump(result,a.out)
    print(len(seen),'photos in',len(scenes),'reviewed scenes')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('groups');b.add_argument('--out',required=True)
    c=sub.add_parser('compile');c.add_argument('groups');c.add_argument('assignments');c.add_argument('--out',required=True)
    a=p.parse_args();build(a) if a.command=='build' else compile_(a)
if __name__=='__main__':main()
