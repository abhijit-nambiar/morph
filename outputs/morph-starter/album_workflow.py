#!/usr/bin/env python3
"""Offline prompt-to-album workflow with global, regional, or face lighting."""
import argparse,json,math,re,subprocess,sys
from pathlib import Path
import morph,prompt_target,face_scene_pipeline


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('groups',nargs='?');p.add_argument('--originals');p.add_argument('--gap-minutes',type=float,default=5.)
    detection=p.add_mutually_exclusive_group();detection.add_argument('--detections');detection.add_argument('--detector')
    p.add_argument('--prompt',required=True);p.add_argument('--out',required=True);p.add_argument('--full',action='store_true');p.add_argument('--preview-samples',type=int,default=3);p.add_argument('--size',type=int,default=320);p.add_argument('--corrections');a=p.parse_args()
    if bool(a.groups)==bool(a.originals):raise ValueError('Provide a group file or --originals')
    if a.size<32 or a.preview_samples<1 or not math.isfinite(a.gap_minutes) or a.gap_minutes<=0:raise ValueError('Invalid size, preview count or gap')
    clauses=[c.strip() for c in re.split(r'[,;]',re.sub(r'\band\b',',',a.prompt.lower()))]
    face_prompt=any(re.fullmatch(r'(?:(?:slightly|strongly) )?(?:brighten|darken) faces',c) for c in clauses)
    if face_prompt:
        face_scene_pipeline.parse_prompt(a.prompt)
        if not (a.detections or a.detector):raise ValueError('Face prompts require --detections or --detector')
        args=[sys.executable,str(Path(__file__).with_name('face_album_workflow.py'))]
        if a.groups:args.append(a.groups)
        else:args+=['--originals',a.originals,'--gap-minutes',str(a.gap_minutes)]
        args+=['--detections',a.detections] if a.detections else ['--detector',a.detector]
        args+=['--prompt',a.prompt,'--out',a.out,'--size',str(a.size),'--preview-samples',str(a.preview_samples)]
        if a.full:args.append('--full')
        if a.corrections:args+=['--corrections',a.corrections]
        subprocess.run(args,check=True);return
    _,_,local=prompt_target.parse_with_local(a.prompt)
    if a.detections or a.detector:raise ValueError('Face detection options are only used for face-light prompts')
    if a.originals:
        folder=Path(a.originals)
        if not folder.is_dir():raise ValueError('Originals directory not found')
        photos=morph.files(folder)
        if not photos:raise ValueError('No supported images in originals directory')
        groups=morph.validate_groups({'groups':morph.chronological_groups(photos,a.gap_minutes)})
    else:groups=morph.validate_groups(json.loads(Path(a.groups).read_text()))
    if a.corrections:
        config=json.loads(Path(a.corrections).read_text())
        if not isinstance(config,dict):raise ValueError('Corrections must be an object')
        local_strengths=config.get('local_strengths',{})
        if not isinstance(local_strengths,dict) or (local_strengths and not local):raise ValueError('Local strengths require regional grid edits')
        if set(local_strengths)-{g['id'] for g in groups}:raise ValueError('Unknown scene in local strengths')
        for value in local_strengths.values():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid local strength')
        common={k:v for k,v in config.items() if k!='local_strengths'}
        if set(common)-{'album','scenes','photos'} or not all(isinstance(v,dict) for v in common.values()):raise ValueError('Invalid correction sections')
        face_scene_pipeline.validate_corrections(common,{g['id']:[(str(Path(p).resolve()),[]) for p in g['photos']] for g in groups})
        import gamut_renderer
        gamut_renderer.vec(common.get('album',{}))
    out=Path(a.out).resolve()
    if out.exists():raise ValueError('Output directory already exists')
    out.mkdir(parents=True);groupfile=Path(a.groups).resolve() if a.groups else out/'groups.json'
    if a.originals:morph.dump({'version':1,'method':'chronological baseline; manually review groups','gap_minutes':a.gap_minutes,'groups':groups},groupfile)
    state={'status':'running','mode':'album' if a.full else 'preview','renderer':'grid' if local else 'gamut','prompt':a.prompt,'groups':str(groupfile),'completed_stages':[]}
    def save():morph.dump(state,out/'run.json')
    def run(stage,script,*args):
        state['active_stage']=stage;save();print('==',stage,flush=True);subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*map(str,args)],check=True);state['completed_stages'].append(stage);state.pop('active_stage',None);save()
    save()
    try:
        run('review-groups','review_groups.py',groupfile,'--out',out/'groups-review')
        fitargs=['album',groupfile,'--preview-only','--target-engine','gaussian-soft' if local else 'pillow-soft','--prompt',a.prompt,'--renderer','grid' if local else 'gamut','--solver','least-squares' if local else 'gauss-newton','--size',a.size,'--out',out/'fits']
        if local:fitargs+=['--regularization','.0001']
        run('fit','morph.py',*fitargs)
        args=['apply',groupfile,out/'fits','--out',out/'album']
        if not a.full:args+=['--preview-samples',a.preview_samples]
        if a.corrections:args+=['--corrections',str(Path(a.corrections).resolve())]
        run('render','morph.py',*args);run('review','review_global_album.py',out/'album'/'review.json','--out',out/'review')
        if a.full:run('verify','verify_global_album.py',groupfile,out/'fits',out/'album','--out',out/'verification.json')
        state['status']='complete';state['outputs']={'review':'review/index.html','groups_review':'groups-review/index.json','diagnostics':'album/review.json','verification':'verification.json' if a.full else None};save();print('Done:',out/'review'/'index.html',flush=True)
    except (Exception,KeyboardInterrupt) as e:state['status']='failed';state['error']=str(e);save();raise
if __name__=='__main__':main()
