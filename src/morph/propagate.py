"""Carry the look of 1..N reference edits to every photo in the album.

For each photo we

1. blend the reference recipes by similarity (same scene > similar setting > nearby in time);
2. predict where each region (face skin, body skin, hair, clothes, subject, background, sky, global)
   *should* land in Lab after editing: the reference's edited value plus a damped share of how this
   photo's original differs from the reference's original (so a darker frame is lifted to match,
   but a genuinely different backdrop keeps some of its character);
3. harmonize those targets within each scene (pull toward the scene consensus) and, more softly,
   across scenes (skin tone and overall key);
4. solve a small per-photo correction (global exposure/WB + per-region exposure/WB/saturation)
   with the differentiable renderer so the rendered region statistics hit the targets.

The final per-photo recipe is purely global/regional sliders - deterministic and non-structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .album import Album, region_stats
from .colorspace import srgb_to_lab
from .recipe import GLOBAL_KEYS, LOCAL_KEYS, LOCAL_REGIONS, Recipe, average
from .render import Params, _soft_clip, render, render_recipe, to_tensor
from .scenes import Scene

STAT_REGIONS = ("global", "face_skin", "body_skin", "hair", "clothes", "subject", "background", "sky")
CORR_GLOBAL = ("exposure", "temperature", "tint", "saturation")
CORR_LOCAL = ("exposure", "temperature", "tint", "saturation")
CORR_REGIONS = ("face_skin", "body_skin", "hair", "clothes", "background", "sky")


@dataclass
class PropagationConfig:
    # share of the original photo-to-reference difference that survives editing (0 = identical output).
    # Defaults tuned by leave-out evaluation on the bundled hand-edited album (see evaluate.py).
    keep_within: dict = field(default_factory=lambda: {"L": 0.15, "ab": 0.09, "skin_L": 0.09, "skin_ab": 0.05})
    keep_across: dict = field(default_factory=lambda: {"L": 0.30, "ab": 0.21, "skin_L": 0.17, "skin_ab": 0.09})
    harmonize_within: float = 0.6  # pull of each photo's targets toward its scene consensus
    harmonize_across: float = 0.35  # pull of skin / global key toward the album consensus
    # how far each photo moves from the plain (similarity-blended) reference recipe toward its predicted target
    pursue_within: float = 0.5
    pursue_across: float = 0.5
    steps: int = 80
    refine_steps: int = 8
    side: int = 256
    clip_weight: float = 60.0  # penalty on highlights pushed past white beyond what the reference edit itself clips
    clip_margin: float = 0.01


@dataclass
class Reference:
    name: str
    recipe: Recipe
    scene: str = ""
    orig_stats: dict = field(default_factory=dict)
    edit_stats: dict = field(default_factory=dict)
    clip_added: float = 0.0  # share of pixels the edit pushes to white on the reference itself


def make_reference(album: Album, name: str, recipe: Recipe, scene: str = "") -> Reference:
    img = album.work_image(name)
    m = album.masks(name).masks
    out = render_recipe(img, m, recipe)
    return Reference(album.get(name).name, recipe, scene, region_stats(img, m), region_stats(out, m), clip_added(img, m, recipe))


def _prep(img: np.ndarray, masks: dict[str, np.ndarray], side: int) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = img.shape[:2]
    s = side / max(h, w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    x = to_tensor(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    mt = torch.zeros((1, len(LOCAL_REGIONS), size[1], size[0]))
    for k, r in enumerate(LOCAL_REGIONS):
        if r in masks:
            mt[0, k] = torch.from_numpy(cv2.resize(masks[r], size, interpolation=cv2.INTER_AREA))
    return x, mt


def clip_added(img: np.ndarray, masks: dict[str, np.ndarray], recipe: Recipe, side: int = 256) -> float:
    x, mt = _prep(img, masks, side)
    with torch.no_grad():
        raw = render(x, mt, Params.from_recipe(recipe), gamut_map=False)
        return max(0.0, float(_overshoot(raw) - _overshoot(x)))


def _mean(stats: dict, r: str) -> np.ndarray | None:
    s = stats.get(r)
    return None if s is None or s["coverage"] < 0.004 else np.array(s["mean"], np.float64)


def reference_weights(album: Album, refs: list[Reference], scenes: list[Scene], temperature: float = 0.08) -> np.ndarray:
    names = album.names
    scene_of = {n: s.id for s in scenes for n in s.photos}
    idx = [names.index(r.name) for r in refs]
    eb = album.emb_background
    W = np.zeros((len(names), len(refs)))
    for i, n in enumerate(names):
        sims = np.array([float(eb[i] @ eb[j]) for j in idx])
        same = np.array([scene_of.get(n) == scene_of.get(r.name) for r in refs], float)
        logits = sims / temperature + same * 4.0
        w = np.exp(logits - logits.max())
        W[i] = w / w.sum()
    return W


def _targets(album: Album, refs: list[Reference], W: np.ndarray, scenes: list[Scene], cfg: PropagationConfig) -> list[dict]:
    scene_of = {n: s.id for s in scenes for n in s.photos}
    out = []
    for i, p in enumerate(album.photos):
        tgt, wsum = {}, {}
        for j, ref in enumerate(refs):
            if W[i, j] < 1e-3:
                continue
            keep = cfg.keep_within if scene_of.get(p.name) == scene_of.get(ref.name) else cfg.keep_across
            for r in STAT_REGIONS:
                o, ro, re_ = _mean(p.stats, r), _mean(ref.orig_stats, r), _mean(ref.edit_stats, r)
                if o is None:
                    continue
                if ro is None or re_ is None:  # reference lacks this region: map through the global shift
                    ro, re_ = _mean(ref.orig_stats, "global"), _mean(ref.edit_stats, "global")
                    if ro is None:
                        continue
                    t = o + (re_ - ro)
                else:
                    skin = r in ("face_skin", "body_skin")
                    kL, kab = (keep["skin_L"], keep["skin_ab"]) if skin else (keep["L"], keep["ab"])
                    t = re_ + np.array([kL, kab, kab]) * (o - ro)
                tgt[r] = tgt.get(r, 0) + W[i, j] * t
                wsum[r] = wsum.get(r, 0) + W[i, j]
        out.append({r: tgt[r] / wsum[r] for r in tgt})
    # within-scene harmonization: pull toward the scene's consensus target
    by_scene: dict[str, list[int]] = {}
    for i, p in enumerate(album.photos):
        by_scene.setdefault(scene_of.get(p.name, "?"), []).append(i)
    h = cfg.harmonize_within
    for idx in by_scene.values():
        for r in STAT_REGIONS:
            vals = [out[i][r] for i in idx if r in out[i]]
            if len(vals) < 2:
                continue
            med = np.median(vals, 0)
            for i in idx:
                if r in out[i]:
                    out[i][r] = (1 - h) * out[i][r] + h * med
    # cross-scene harmonization: skin tone and overall key converge softly album-wide
    c = cfg.harmonize_across
    for r in ("face_skin", "body_skin", "global"):
        vals = [o[r] for o in out if r in o]
        if len(vals) < 2:
            continue
        med = np.median(vals, 0)
        wvec = np.array([c * 0.6, c, c]) if r == "global" else np.array([c, c, c])
        for o in out:
            if r in o:
                o[r] = (1 - wvec) * o[r] + wvec * med
    return out


def _overshoot(raw: torch.Tensor) -> torch.Tensor:
    """Soft share of pixels whose brightest channel is pushed to or past white."""
    return torch.sigmoid((raw.amax(1) - 1.0) / 0.02).mean()


def _correct(img: np.ndarray, masks: dict[str, np.ndarray], base: Recipe, targets: dict, cfg: PropagationConfig, gain: float = 1.0,
             clip_budget: float | None = None) -> Recipe:
    """``clip_budget``: how much clipping the references' own edits add; the photo may keep at most that much on top of its own."""
    x, mt = _prep(img, masks, cfg.side)
    stat_w = {}
    for r in targets:
        if r == "global":
            stat_w[r] = torch.ones((1, 1) + x.shape[-2:])
        elif r in LOCAL_REGIONS:
            stat_w[r] = mt[:, LOCAL_REGIONS.index(r) : LOCAL_REGIONS.index(r) + 1]
    stat_w = {r: m for r, m in stat_w.items() if float(m.mean()) > 0.004}
    if not stat_w:
        return base
    importance = {"face_skin": 3.0, "body_skin": 1.5, "global": 2.0, "subject": 1.0, "background": 1.0, "hair": 0.7, "clothes": 0.7, "sky": 0.5}
    tg = {r: torch.tensor(targets[r], dtype=torch.float32) for r in stat_w}

    p0 = Params.from_recipe(base)
    gd = torch.zeros(len(CORR_GLOBAL), requires_grad=True)
    regs = [r for r in CORR_REGIONS if float(mt[0, LOCAL_REGIONS.index(r)].mean()) > 0.004]
    ld = torch.zeros((len(regs), len(CORR_LOCAL)), requires_grad=True)
    gi = [GLOBAL_KEYS.index(k) for k in CORR_GLOBAL]
    li = [LOCAL_KEYS.index(k) for k in CORR_LOCAL]
    ri = [LOCAL_REGIONS.index(r) for r in regs]
    ag = {k for k in GLOBAL_KEYS if abs(base.g(k)) > 1e-4} | set(CORR_GLOBAL)
    al = {LOCAL_REGIONS.index(r): {k for k in LOCAL_KEYS if abs(base.l(r, k)) > 1e-4} for r in LOCAL_REGIONS}
    for k in ri:
        al[k] = al[k] | set(CORR_LOCAL)
    al = {k: v for k, v in al.items() if v}
    opt = torch.optim.Adam([gd, ld], lr=0.04)
    pz = Params.from_recipe(Recipe())
    corr_local = {k: set(CORR_LOCAL) for k in ri}
    with torch.no_grad():
        xb_raw = render(x, mt, p0, active_global=ag, active_local=al, gamut_map=False)
        clip0 = _overshoot(xb_raw)
        if clip_budget is not None:
            clip0 = torch.minimum(clip0, _overshoot(x) + clip_budget + cfg.clip_margin)
        xb = _soft_clip(xb_raw)
        lab_b = srgb_to_lab(xb.clamp(0, 1).permute(0, 2, 3, 1))[0]
        for r, m in stat_w.items():  # only pursue part of the way from the plain recipe copy toward the target
            mm = m[0, 0][..., None]
            mu_b = (lab_b * mm).sum((0, 1)) / mm.sum()
            tg[r] = mu_b + gain * (tg[r] - mu_b)

    def params(p):
        g = p.gv.clone()
        g[gi] = g[gi] + gd
        l = p.lv.clone()
        for a, k in enumerate(ri):
            l[k, li] = l[k, li] + ld[a]
        return Params(g, l)

    def loss_of(raw):
        out = _soft_clip(raw)
        lab = srgb_to_lab(out.clamp(0, 1).permute(0, 2, 3, 1))[0]
        loss = cfg.clip_weight * (_overshoot(raw) - clip0).clamp(min=0)
        for r, m in stat_w.items():
            mm = m[0, 0][..., None]
            mu = (lab * mm).sum((0, 1)) / mm.sum()
            d = (mu - tg[r]) / torch.tensor([4.0, 3.0, 3.0])
            loss = loss + importance.get(r, 1.0) * (d**2).sum()
        return loss + 0.6 * (gd**2).sum() + 0.3 * (ld**2).sum()

    # Stage 1: solve the correction cheaply on top of the base render; stage 2: refine through the full recipe.
    for step in range(cfg.steps + cfg.refine_steps):
        opt.zero_grad()
        if step < cfg.steps:
            out = render(xb, mt, params(pz), active_global=set(CORR_GLOBAL), active_local=corr_local, gamut_map=False)
        else:
            out = render(x, mt, params(p0), active_global=ag, active_local=al, gamut_map=False)
        loss_of(out).backward()
        opt.step()
    rec = params(p0).to_recipe(dict(base.meta)).clamped().pruned()
    rec.meta["correction"] = {"global": dict(zip(CORR_GLOBAL, gd.detach().numpy().round(4).tolist())),
                              "local": {r: dict(zip(CORR_LOCAL, ld[a].detach().numpy().round(4).tolist())) for a, r in enumerate(regs)}}
    return rec


def propagate(album: Album, refs: list[Reference], scenes: list[Scene], cfg: PropagationConfig | None = None, progress=None) -> dict[str, Recipe]:
    cfg = cfg or PropagationConfig()
    W = reference_weights(album, refs, scenes)
    targets = _targets(album, refs, W, scenes, cfg)
    ref_names = {r.name: r for r in refs}
    scene_of = {n: s.id for s in scenes for n in s.photos}
    out = {}
    for i, p in enumerate(album.photos):
        if p.name in ref_names:
            rec = ref_names[p.name].recipe.copy()
            rec.meta = {**rec.meta, "role": "reference"}
        else:
            base = average([r.recipe for r in refs], W[i].tolist())
            base.meta = {"weights": {r.name: round(float(W[i, j]), 4) for j, r in enumerate(refs) if W[i, j] > 1e-3}}
            same = np.array([scene_of.get(p.name) == r.scene for r in refs], float)
            gain = float(W[i] @ (same * cfg.pursue_within + (1 - same) * cfg.pursue_across))
            budget = float(W[i] @ np.array([r.clip_added for r in refs]))
            rec = _correct(album.work_image(p.name), album.masks(p.name).masks, base, targets[i], cfg, gain, budget)
        out[p.name] = rec
        if progress:
            progress(i + 1, len(album.photos), p.name)
    return out
