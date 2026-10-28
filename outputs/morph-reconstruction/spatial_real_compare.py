#!/usr/bin/env python3
"""Compare global and radial mappings on real held-out same-size pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import morph
import poly_color
import spatial_light


def rmse(x, y):
    return float(np.sqrt(np.mean((x-y)**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', help='gamut-real-transfer.json; supplies the same scene splits')
    p.add_argument('edits')
    p.add_argument('--out', required=True)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--regularization', type=float, default=.001)
    a = p.parse_args()
    baseline = json.loads(Path(a.baseline).read_text())
    scenes, rows = [], []
    for scene in baseline['scenes']:
        if 'recipe' not in scene:
            continue
        rep = scene['representative']
        x = morph.load(rep, a.size)
        y = morph.load(Path(a.edits)/Path(rep).name, a.size)
        wp, _ = poly_color.fit(x, y, regularization=a.regularization)
        ws = spatial_light.fit(x, y, regularization=a.regularization)
        held = []
        for row in scene['heldout']:
            photo = row['photo']
            x = morph.load(photo, a.size)
            y = morph.load(Path(a.edits)/Path(photo).name, a.size)
            result = {'photo': photo, 'source_rmse': rmse(x,y),
                      'polynomial_rmse': rmse(poly_color.render(x,wp),y),
                      'spatial_rmse': rmse(spatial_light.render(x,ws),y)}
            held.append(result)
            rows.append(result)
        scenes.append({'id': scene['id'], 'representative': rep,
                       'polynomial_mapping': poly_color.serialize(wp),
                       'spatial_mapping': spatial_light.serialize(ws), 'heldout': held})
        print(scene['id'], len(held), flush=True)
    if not rows:
        raise ValueError('No held-out photos in baseline')
    poly = np.array([r['polynomial_rmse'] for r in rows]); spatial = np.array([r['spatial_rmse'] for r in rows]); source = np.array([r['source_rmse'] for r in rows])
    morph.dump({'scope': 'One album; chronological groups not validated; same-size pairs may be misaligned or edited with different intentions. One representative per group; other photos held out. Both methods use the same fitting pixels and regularization. Pixel RMSE is not perceived quality. Mask is frame-relative.',
        'size': a.size, 'regularization': a.regularization,
        'summary': {'scenes': len(scenes), 'heldout_photos':len(rows), 'spatial_lower_than_polynomial':int(np.sum(spatial < poly)), 'spatial_lower_than_source':int(np.sum(spatial < source)), 'polynomial_lower_than_source':int(np.sum(poly < source)), 'source_mean_rmse':float(source.mean()), 'polynomial_mean_rmse':float(poly.mean()), 'spatial_mean_rmse':float(spatial.mean()), 'polynomial_median_rmse':float(np.median(poly)), 'spatial_median_rmse':float(np.median(spatial))}, 'scenes': scenes}, a.out)

if __name__ == '__main__':
    main()
