#!/usr/bin/env python3
"""Compare transferable polynomial color mappings against bounded slider recipes."""
import argparse,json
from pathlib import Path
import numpy as np
import morph,poly_color,gamut_renderer

def main():
    p=argparse.ArgumentParser();p.add_argument('gamut_report');p.add_argument('edits');p.add_argument('--out',required=True);p.add_argument('--regularization',type=float,default=.001);p.add_argument('--size',type=int,default=256);a=p.parse_args();data=json.loads(Path(a.gamut_report).read_text());scenes=[];rows=[]
    for scene in data['scenes']:
        if 'recipe' not in scene:continue
        rep=scene['representative'];target=Path(a.edits)/Path(rep).name;x=morph.load(rep,a.size);y=morph.load(target,a.size);w,metrics=poly_color.fit(x,y,regularization=a.regularization);bounded=gamut_renderer.vec(scene['recipe']);held=[]
        for r in scene['heldout']:
            x=morph.load(r['photo'],a.size);y=morph.load(Path(a.edits)/Path(r['photo']).name,a.size);v=float(np.sqrt(np.mean((poly_color.render(x,w)-y)**2)));entry={'photo':r['photo'],'bounded_rmse':r['selective_rmse'],'polynomial_rmse':v,'bounded_clipping':morph.clipping_diagnostic(x,bounded,renderer=gamut_renderer.render),'polynomial_clipping':morph.clipping_diagnostic(x,w,renderer=poly_color.render)};rows.append(entry);held.append(entry)
        scenes.append({'id':scene['id'],'representative':rep,'mapping':poly_color.serialize(w),'fit_metrics':metrics,'heldout':held});print(scene['id'],len(held),flush=True)
    b=np.array([r['bounded_rmse'] for r in rows]);c=np.array([r['polynomial_rmse'] for r in rows]);morph.dump({'scope':'Same album and proposed groups; held-out photos excluded from each representative fit; input/edit pairs may be misaligned. More flexible polynomial mapping is global, not semantic or spatial, and RGB outputs can clip. Methods fit different pixel sample sizes.','regularization':a.regularization,'summary':{'heldout':len(rows),'polynomial_lower_error':int(np.sum(c<b)),'bounded_mean_rmse':float(b.mean()),'polynomial_mean_rmse':float(c.mean()),'bounded_median_rmse':float(np.median(b)),'polynomial_median_rmse':float(np.median(c)),'bounded_clipping_flags':sum(r['bounded_clipping']['flag'] for r in rows),'polynomial_clipping_flags':sum(r['polynomial_clipping']['flag'] for r in rows)},'scenes':scenes},a.out)
if __name__=='__main__':main()
