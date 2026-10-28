#!/usr/bin/env python3
"""Run the experimental offline face-aware album workflow."""
import argparse,json,math,subprocess,sys
from pathlib import Path
import morph,face_scene_pipeline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups',nargs='?',help='Reviewed scene group JSON')
    p.add_argument('--originals',help='Original image folder to group chronologically')
    p.add_argument('--gap-minutes',type=float,default=5.,help='Capture-time gap for proposed chronological groups')
    detections=p.add_mutually_exclusive_group(required=True)
    detections.add_argument('--detections',help='Checksum-verified detection JSON')
    detections.add_argument('--detector',help='Compiled local face detector executable')
    p.add_argument('--prompt',required=True,help='Supported offline prompt, including brighten faces or darken faces')
    p.add_argument('--out',required=True,help='New output directory')
    p.add_argument('--full',action='store_true',help='Render all photos; default is preview samples only')
    p.add_argument('--preview-samples',type=int,default=3,help='Full-resolution photos per scene in preview mode')
    p.add_argument('--corrections',help='Optional reviewed album/scene/photo corrections JSON')
    p.add_argument('--size',type=int,default=320,help='Maximum fitting dimension')
    a=p.parse_args();face_scene_pipeline.parse_prompt(a.prompt)
    if bool(a.groups)==bool(a.originals):raise ValueError('Provide either a group file or --originals')
    if a.size<32 or a.preview_samples<1 or not math.isfinite(a.gap_minutes) or a.gap_minutes<=0:raise ValueError('Invalid size, preview samples or capture-time gap')
    if a.originals:
        folder=Path(a.originals)
        if not folder.is_dir():raise ValueError('Originals directory not found: '+str(folder))
        photos=morph.files(folder)
        if not photos:raise ValueError('No supported images in originals directory')
        groups=morph.validate_groups({'groups':morph.chronological_groups(photos,a.gap_minutes)})
    else:groups=morph.validate_groups(json.loads(Path(a.groups).read_text()))
    if a.corrections:
        config=face_scene_pipeline.load_corrections(a.corrections)
        face_scene_pipeline.validate_corrections(config,{g['id']:[(str(Path(p).resolve()),[]) for p in g['photos']] for g in groups})
    inputs=([Path(a.groups)] if a.groups else [])+[Path(photo) for group in groups for photo in group['photos']]
    for name in (a.detections,a.detector,a.corrections):
        if name:
            path=Path(name)
            if not path.is_file():raise ValueError('Input file not found: '+str(path))
            inputs.append(path)
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Output directory already exists')
    if any(path.resolve()==out for path in inputs):raise ValueError('Output would overwrite an input')
    out.mkdir(parents=True)
    group_path=Path(a.groups).resolve() if a.groups else out/'groups.json'
    if a.originals:morph.dump({'version':1,'method':'chronological baseline; manually review groups','gap_minutes':a.gap_minutes,'groups':groups},group_path)
    state={'status':'running','mode':'album' if a.full else 'preview','prompt':a.prompt,'groups':str(group_path),'groups_origin':'proposed chronological' if a.originals else 'supplied','completed_stages':[]}
    def save():morph.dump(state,out/'run.json')
    def run(stage,script,*args):
        state['active_stage']=stage;save();print('==',stage,flush=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*map(str,args)],check=True)
        state['completed_stages'].append(stage);state.pop('active_stage',None);save()
    save()
    try:
        run('review-groups','review_groups.py',group_path,'--out',out/'groups-review')
        if a.detector:
            detection_path=out/'detections.json'
            run('detect','detect_scene_faces.py',group_path,'--detector',a.detector,'--out',detection_path)
        else:detection_path=Path(a.detections).resolve()
        run('review-detections','review_face_detections.py',detection_path,'--out',out/'detections-review')
        run('fit','face_scene_pipeline.py','fit-album',group_path,detection_path,'--prompt',a.prompt,'--size',a.size,'--out',out/'fits')
        args=['apply-album',out/'fits','--out',out/'album']
        if not a.full:args.extend(['--preview-samples',a.preview_samples])
        if a.corrections:args.extend(['--corrections',Path(a.corrections).resolve()])
        run('render','face_scene_pipeline.py',*args)
        run('review','review_face_album.py',out/'album','--out',out/'review')
        run('clipping-audit','audit_face_album.py',out/'fits',out/'album','--out',out/'clipping-audit.json')
        run('evaluate-transfer','evaluate_face_transfer.py',out/'fits',out/'album','--out',out/'transfer-evaluation.json')
        run('scene-consistency','audit_scene_consistency.py',out/'fits',out/'album','--out',out/'scene-consistency.json')
        if a.full:run('verify','verify_face_album.py',out/'fits',out/'album','--out',out/'verification.json')
        state['status']='complete';state['outputs']={'review':'review/index.html','groups_review':'groups-review/index.json','detections_review':'detections-review/index.json','clipping_audit':'clipping-audit.json','transfer_evaluation':'transfer-evaluation.json','scene_consistency':'scene-consistency.json','verification':'verification.json' if a.full else None};save()
        print('Done:',out/'review'/'index.html',flush=True)
    except (Exception,KeyboardInterrupt) as e:
        state['status']='failed';state['error']=str(e);save();raise
if __name__=='__main__':main()
