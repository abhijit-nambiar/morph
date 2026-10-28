#!/usr/bin/env python3
"""Evaluate transfer of a recipe fitted from a human-edited pair to other pairs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import morph

def rmse(x,y): return float(np.sqrt(np.mean((x-y)**2)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('groups');ap.add_argument('edits');ap.add_argument('--out',required=True);ap.add_argument('--size',type=int,default=256);ap.add_argument('--solver',choices=['pattern','gauss-newton'],default='pattern');ap.add_argument('--iterations',type=int,default=45);args=ap.parse_args()
    groups=morph.validate_groups(json.loads(Path(args.groups).read_text()));edits=Path(args.edits);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    scenes=[];sheets=[]
    for group in groups:
        pairs=[]; skipped=[]
        for p in group['photos']:
            src=Path(p);dst=edits/src.name
            if not dst.exists():skipped.append({'photo':p,'reason':'missing edit'});continue
            with Image.open(src) as x,Image.open(dst) as y:
                if morph.oriented_size(x)!=morph.oriented_size(y):skipped.append({'photo':p,'reason':'dimensions differ; alignment required'});continue
            pairs.append((src,dst))
        if not pairs:scenes.append({'id':group['id'],'status':'no same-size pairs','skipped':skipped});continue
        rep=next((p for p in pairs if str(p[0])==group['representative']),pairs[0]);x=morph.load(rep[0],args.size);y=morph.load(rep[1],args.size);fit,fit_metrics=morph.run_fit(x,y,iterations=args.iterations,method=args.solver)
        results=[];visuals=[]
        for src,dst in pairs:
            if src==rep[0]:continue
            image=morph.load(src,args.size);target=morph.load(dst,args.size);out_image=morph.render(image,fit)
            results.append({'photo':str(src),'edit':str(dst),'identity_rmse':rmse(image,target),'transfer_rmse':rmse(out_image,target)})
            visuals.append((src.name,(image,target,out_image)))
        scene_dir=out/group['id'];scene_dir.mkdir(exist_ok=True); pages=morph.comparison_sheet(visuals,scene_dir)
        scenes.append({'id':group['id'],'configured_representative':group['representative'],'fitting_representative':str(rep[0]),'fitted_recipe':morph.recipe(fit),'fit_metrics':fit_metrics,'heldout_results':results,'skipped':skipped,'comparison_sheets':pages})
        print(group['id'],'held-out',len(results),'skipped',len(skipped),flush=True)
    held=[r for s in scenes for r in s.get('heldout_results',[])];before=np.array([r['identity_rmse'] for r in held]);after=np.array([r['transfer_rmse'] for r in held]);
    summary={'heldout_pairs':len(held),'improved_vs_original':int(np.sum(after<before)),'median_identity_rmse':float(np.median(before)) if len(before) else None,'median_transfer_rmse':float(np.median(after)) if len(after) else None}
    morph.dump({'solver':args.solver,'iterations':args.iterations,'caveats':['Groups are unvalidated; same dimensions do not imply pixel alignment; per-photo human edits may differ intentionally.','Identity is a useful baseline but RMSE is not a measure of professional quality.'],'summary':summary,'scenes':scenes},out/'report.json')
if __name__=='__main__':main()
