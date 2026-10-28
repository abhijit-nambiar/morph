#!/usr/bin/env python3
"""Render selected held-out examples using mappings in a spatial comparison report."""
import argparse
import json
import shutil
from pathlib import Path
import morph
import poly_color
import spatial_light


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report')
    p.add_argument('edits')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    report = json.loads(Path(a.report).read_text())
    rows = [(r['spatial_rmse']-r['polynomial_rmse'], s, r) for s in report['scenes'] for r in s['heldout']]
    selected = [min(rows, key=lambda x:x[0]), max(rows, key=lambda x:x[0])]
    notes = ['# Full-resolution real-pair examples', '', 'Each folder contains the representative source and its human edit used to fit the mapping, followed by a different held-out scene photo, its human edit, and two recovered edits applied to that held-out source.', '', 'The images are full-resolution JPEGs. Fitting and RMSE comparison used downsized images before JPEG encoding. Selection used the most favorable and least favorable spatial-versus-polynomial error differences in this album; it is not a representative sample. Groups and pair alignment are unverified, and human edits can vary by photo. The spatial mapping uses a frame-relative light mask.', '']
    for name,(diff,scene,row) in zip(('spatial-best','spatial-worst'),selected):
        folder = out/name
        folder.mkdir()
        rep = Path(scene['representative']); photo = Path(row['photo'])
        for label, path in [('representative-original.jpg',rep),('representative-human-edit.jpg',Path(a.edits)/rep.name),('heldout-original.jpg',photo),('heldout-human-edit.jpg',Path(a.edits)/photo.name)]:
            shutil.copy2(path, folder/label)
        x = morph.load(photo)
        for label,renderer,weights in [('heldout-polynomial.jpg',poly_color.render,poly_color.deserialize(scene['polynomial_mapping'])),('heldout-spatial.jpg',spatial_light.render,spatial_light.deserialize(scene['spatial_mapping']))]:
            morph.save_render(x,weights,folder/label,renderer=renderer,source=photo)
        notes.extend([f"## {name}", '', f"Scene: {scene['id']}. Representative: {rep.name}. Held-out photo: {photo.name}.", f"At fitting resolution, source RMSE {row['source_rmse']:.5f}; polynomial {row['polynomial_rmse']:.5f}; spatial {row['spatial_rmse']:.5f}.", ''])
        print(name, photo.name, flush=True)
    (out/'README.md').write_text('\n'.join(notes))

if __name__ == '__main__':
    main()
