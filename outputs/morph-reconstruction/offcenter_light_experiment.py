#!/usr/bin/env python3
"""Test a centered-light reconstruction against independent off-center lighting targets."""
import argparse
import json
from pathlib import Path
import numpy as np
import morph
import poly_color
import spatial_light


def target(x, settings, center, strength):
    """Off-center Gaussian illumination after independent tone/color operations."""
    h, w = x.shape[:2]
    yy = (np.arange(h, dtype=np.float32)+.5)/h
    xx = (np.arange(w, dtype=np.float32)+.5)/w
    mask = np.exp(-.5*(((xx[None,:]-center[0])/.18)**2 + ((yy[:,None]-center[1])/.23)**2))
    base = morph.pillow_soft_target(x, 'neutral', settings)
    return np.clip(base + strength*mask[...,None]*x*(1-x), 0, 1)


def rmse(x, y):
    return float(np.sqrt(np.mean((x-y)**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('groups'); p.add_argument('settings'); p.add_argument('--out', required=True)
    p.add_argument('--size', type=int, default=256); p.add_argument('--heldout', type=int, default=2)
    a = p.parse_args()
    groups = morph.validate_groups(json.loads(Path(a.groups).read_text()))
    settings = morph.soft_settings('neutral', json.loads(Path(a.settings).read_text()))
    rows = []
    # Both masks lie away from the fixed centered mask; settings and strength are predetermined.
    for g in groups:
        held = [p for p in g['photos'] if p != g['representative']][:a.heldout]
        if not held: continue
        original = morph.load(g['representative'], a.size)
        photos = [(str(path), morph.load(path, a.size)) for path in held]
        for center in ((.2,.3),(.8,.7)):
            for strength in (-.3,.3):
                want = target(original, settings, center, strength)
                wp, _ = poly_color.fit(original, want, regularization=.0001)
                ws = spatial_light.fit(original, want, regularization=.0001)
                for path,x in photos:
                    y = target(x, settings, center, strength)
                    rows.append({'scene':g['id'], 'representative':str(g['representative']), 'photo':path,
                                 'center':list(center), 'strength':strength, 'source_rmse':rmse(x,y),
                                 'global_rmse':rmse(poly_color.render(x,wp),y),
                                 'centered_rmse':rmse(spatial_light.render(x,ws),y)})
        print(g['id'],len(held),'heldout',flush=True)
    if not rows: raise ValueError('No held-out photos')
    summary={}
    for center in ((.2,.3),(.8,.7)):
        for strength in (-.3,.3):
            subset=[r for r in rows if r['center']==list(center) and r['strength']==strength]
            g=np.array([r['global_rmse'] for r in subset]); s=np.array([r['centered_rmse'] for r in subset])
            summary[f'{center[0]},{center[1]}:{strength}']={'heldout':len(subset),'centered_lower':int(np.sum(s<g)), 'global_mean_rmse':float(g.mean()), 'centered_mean_rmse':float(s.mean())}
    morph.dump({'scope':'Synthetic off-center Gaussian light masks plus independent Pillow color/tone targets, same fixed mask per photo within each case; one album with proposed groups, one representative and up to two held-out photos per group. Same sampled pixels and regularization for both fits. Does not test subject-aware transfer, real edits, or photographic quality.', 'settings':settings, 'fit_size':a.size,'regularization':.0001,'summary':summary,'cases':rows},a.out)

if __name__ == '__main__': main()
