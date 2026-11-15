"""Benchmark against hand-edited albums: use K hand edits as references, predict the rest.

Reports, per photo and aggregated, CIEDE2000 between the prediction and the photographer's edit
(measured only where the edit is structurally comparable to the original), for:

* ``original``   - doing nothing,
* ``copy``       - pasting the nearest reference's recipe unchanged (what a preset/sync does),
* ``morph``      - full propagation with lighting adaptation and harmonization.

Also reports within-scene consistency (spread of face-skin / global Lab across a scene).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .album import ANALYSIS_VERSION, Album
from .colorspace import delta_e2000, srgb_to_lab
from .fit import align, fit_recipe, structure_confidence
from .imageio import list_images, load_rgb, save_png_u8
from .propagate import PropagationConfig, make_reference, propagate, reference_weights
from .recipe import average
from .render import render_recipe
from .scenes import cluster, suggest


def _aligned_edits(album: Album, edits_dir: Path, cache: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache.mkdir(parents=True, exist_ok=True)
    edits = {p.stem: p for p in list_images(edits_dir)}
    out = {}
    for p in album.photos:
        e = edits.get(p.stem)
        if e is None:
            continue
        st = e.stat()
        sig = hashlib.sha1(f"{ANALYSIS_VERSION}:{p.sha256}:{e.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:10]
        ft, fw = cache / f"{p.stem}.{sig}.png", cache / f"{p.stem}.{sig}.w.npy"
        if ft.exists() and fw.exists():
            out[p.name] = (load_rgb(ft), np.load(fw).astype(np.float32))
            continue
        img = album.work_image(p.name)
        al = align(img, load_rgb(e, max_side=int(max(img.shape[:2]) * 1.25)))
        w = (structure_confidence(img, al.target) > 0.3).astype(np.float32) * al.valid
        save_png_u8(al.target, ft)
        np.save(fw, w.astype(np.float16))
        out[p.name] = (al.target, w.astype(np.float32))
    return out


def _de(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    d = delta_e2000(srgb_to_lab(a), srgb_to_lab(b))
    return float((d * w).sum() / max(w.sum(), 1))


def evaluate(originals: str | Path, edits: str | Path, out: str | Path, k: int = 3, cache: str | Path | None = None,
             refs: list[str] | None = None, progress=None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    album = Album(originals, cache).analyze(progress=progress)
    scenes = cluster(album)
    targets = _aligned_edits(album, Path(edits), album.cache / "aligned-edits")
    pool = [s for s in suggest(album, scenes, k=len(album.photos)) if s.name in targets]
    ref_names = refs or [s.name for s in pool[:k]]
    references = []
    for n in ref_names:
        img = album.work_image(n)
        t, w = targets[n]
        res = fit_recipe(img, t, album.masks(n).masks, w)
        references.append(make_reference(album, n, res.recipe, next(s.id for s in scenes if n in s.photos)))
    recipes = propagate(album, references, scenes, PropagationConfig(), progress=progress)
    W = reference_weights(album, references, scenes)
    rows = []
    for i, p in enumerate(album.photos):
        if p.name not in targets:
            continue
        img, m = album.work_image(p.name), album.masks(p.name).masks
        t, w = targets[p.name]
        nearest = references[int(np.argmax(W[i]))].recipe
        pred = render_recipe(img, m, recipes[p.name])
        copy = render_recipe(img, m, nearest)
        rows.append({"photo": p.name, "reference": p.name in ref_names,
                     "scene": next(s.id for s in scenes if p.name in s.photos),
                     "original": round(_de(img, t, w), 3), "copy": round(_de(copy, t, w), 3), "morph": round(_de(pred, t, w), 3)})
        if p.name not in ref_names and i % 3 == 0:
            strip = np.concatenate([img, pred, t], 1)
            save_png_u8(strip[:: 2, :: 2], out / "strips" / f"{p.stem}.png")
    held = [r for r in rows if not r["reference"]]
    summary = {k_: round(float(np.mean([r[k_] for r in held])), 3) for k_ in ("original", "copy", "morph")}
    summary.update(references=ref_names, n_photos=len(rows), n_heldout=len(held), scenes=len(scenes))
    summary["morph_beats_copy"] = round(float(np.mean([r["morph"] < r["copy"] for r in held])), 3)
    (out / "evaluation.json").write_text(json.dumps({"summary": summary, "photos": rows}, indent=1))
    return summary
