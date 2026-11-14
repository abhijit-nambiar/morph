"""Reverse-engineer a deterministic recipe from an (original, target) pair.

The target may come from a generative model (which can hallucinate or shift content) or from a
hand edit (which may be cropped/straightened).  We therefore (1) geometrically align the target to
the original, (2) build a per-pixel confidence map that ignores pixels whose local structure
changed, and (3) fit only pixel-wise tone/colour/detail sliders, so whatever is learned is
guaranteed to be non-structural when applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .colorspace import delta_e76, srgb_to_lab
from .recipe import BOUNDS, GLOBAL_KEYS, LOCAL_BOUNDS, LOCAL_KEYS, LOCAL_REGIONS, Recipe
from .render import Params, blur, render, to_tensor

FIT_GLOBAL = tuple(k for k in GLOBAL_KEYS if k not in ("grain", "vignette", "sharpen"))
FIT_LOCAL = tuple(k for k in LOCAL_KEYS if k not in ("dehaze",))
DETAIL = {"clarity", "texture", "dehaze", "noise_reduction", "sharpen"}


@dataclass
class Alignment:
    target: np.ndarray  # target warped into the original's frame
    valid: np.ndarray  # HxW float, 1 where the warped target has content
    matrix: np.ndarray | None = None
    method: str = "identity"
    structural_change: float = 0.0  # fraction of pixels whose structure differs


def _gray_u8(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def align(orig: np.ndarray, target: np.ndarray) -> Alignment:
    """Warp ``target`` onto ``orig``'s pixel grid (handles crops, straightening, rescaling)."""
    H, W = orig.shape[:2]
    if target.shape[:2] == (H, W):
        conf = structure_confidence(orig, target)
        if float((conf > 0.5).mean()) > 0.6:
            return Alignment(target, np.ones((H, W), np.float32), np.eye(3), "identity", float((conf < 0.5).mean()))
    a, b = _gray_u8(orig), _gray_u8(target)
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    M = None
    if da is not None and db is not None and len(ka) > 8 and len(kb) > 8:
        matches = cv2.BFMatcher().knnMatch(db, da, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) >= 12:
            src = np.float32([kb[m.queryIdx].pt for m in good])
            dst = np.float32([ka[m.trainIdx].pt for m in good])
            M, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if M is not None and inl.sum() < 10:
                M = None
    method = "homography"
    if M is None:
        M = _ecc(a, b)
        method = "ecc"
    if M is None:
        warped = cv2.resize(target, (W, H), interpolation=cv2.INTER_AREA)
        valid = np.ones((H, W), np.float32)
        method = "resize"
    else:
        warped = cv2.warpPerspective(target, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        valid = cv2.warpPerspective(np.ones(target.shape[:2], np.float32), M, (W, H), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    conf = structure_confidence(orig, warped) * valid
    return Alignment(warped.astype(np.float32), valid.astype(np.float32), M, method, float(((conf < 0.5) & (valid > 0)).mean()))


def _ecc(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Intensity-based affine registration for low-texture images; returns target->orig 3x3."""
    H, W = a.shape
    sx, sy = W / b.shape[1], H / b.shape[0]
    br = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a.astype(np.float32), br.astype(np.float32), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        return None
    W3 = np.vstack([warp, [0, 0, 1]]).astype(np.float64)
    S_inv = np.diag([1 / sx, 1 / sy, 1.0])
    return np.linalg.inv(S_inv @ W3)


def structure_confidence(orig: np.ndarray, target: np.ndarray, win: int | None = None) -> np.ndarray:
    """Local normalised cross-correlation of luminance detail; ~1 where structure is preserved."""
    H, W = orig.shape[:2]
    win = win or max(5, int(round(max(H, W) / 48)) | 1)

    def hp(img):
        g = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
        return g - cv2.GaussianBlur(g, (0, 0), win / 2)

    a, b = hp(orig), hp(target)
    k = (win, win)
    mab = cv2.blur(a * b, k)
    va = cv2.blur(a * a, k)
    vb = cv2.blur(b * b, k)
    ncc = mab / np.sqrt(va * vb + 1e-8)
    flat = (va < 1e-5) & (vb < 1e-5)  # both flat: nothing to disagree about
    ncc = np.where(flat, 1.0, ncc)
    return np.clip((ncc - 0.15) / 0.45, 0, 1).astype(np.float32)


@dataclass
class FitResult:
    recipe: Recipe
    de_before: float
    de_after: float
    region_de: dict[str, float] = field(default_factory=dict)


def _lab(x: torch.Tensor) -> torch.Tensor:
    return srgb_to_lab(x.clamp(0, 1).permute(0, 2, 3, 1))


def fit_recipe(
    orig: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray] | None,
    weight: np.ndarray | None = None,
    *,
    init: Recipe | None = None,
    steps: int = 260,
    regions: tuple[str, ...] | None = None,
    reg: float = 2e-3,
    max_side: int = 384,
    fit_detail: bool = True,
) -> FitResult:
    """Fit global + per-region sliders so ``render(orig) ~= target`` (all inputs share a grid)."""
    torch.manual_seed(0)
    H, W = orig.shape[:2]
    s = min(1.0, max_side / max(H, W))
    size = (max(1, round(W * s)), max(1, round(H * s)))
    rs = lambda a, interp=cv2.INTER_AREA: cv2.resize(a, size, interpolation=interp) if s < 1 else a
    x = to_tensor(rs(orig).astype(np.float32))
    t = to_tensor(rs(target).astype(np.float32))
    w = torch.from_numpy(rs(np.asarray(weight if weight is not None else np.ones((H, W)), np.float32)))[None, None]
    regions = tuple(r for r in (regions or LOCAL_REGIONS) if masks is not None and r in masks)
    m = torch.zeros((1, len(LOCAL_REGIONS)) + x.shape[-2:])
    cover = {}
    for r in regions:
        ri = LOCAL_REGIONS.index(r)
        m[0, ri] = torch.from_numpy(rs(masks[r].astype(np.float32)))
        cover[r] = float((m[0, ri] * w[0, 0]).mean())
    regions = tuple(r for r in regions if cover[r] > 1.5e-3)
    gkeys = set(FIT_GLOBAL) - (set() if fit_detail else DETAIL)
    lkeys = set(FIT_LOCAL) - (set() if fit_detail else DETAIL)
    al = {LOCAL_REGIONS.index(r): lkeys for r in regions}
    scale = None  # spatial filters scale with the rendered grid (resolution independent)

    p = Params.from_recipe(init or Recipe(), requires_grad=True)
    gmask = torch.tensor([k in gkeys for k in GLOBAL_KEYS], dtype=torch.float32)
    lmask = torch.zeros_like(p.lv)
    for ri in al:
        lmask[ri] = torch.tensor([k in lkeys for k in LOCAL_KEYS], dtype=torch.float32)
    greg = torch.tensor([reg * (4 if k in DETAIL else 1) for k in GLOBAL_KEYS])
    lreg = torch.tensor([reg * (3 if k in DETAIL else 1.5) for k in LOCAL_KEYS])
    glo = torch.tensor([BOUNDS[k][0] for k in GLOBAL_KEYS]); ghi = torch.tensor([BOUNDS[k][1] for k in GLOBAL_KEYS])
    llo = torch.tensor([LOCAL_BOUNDS[k][0] for k in LOCAL_KEYS]); lhi = torch.tensor([LOCAL_BOUNDS[k][1] for k in LOCAL_KEYS])

    tl = _lab(t)
    tlb = _lab(blur(t, 3.0))
    wsum = w.sum().clamp(min=1)
    rw = [(m[:, LOCAL_REGIONS.index(r) : LOCAL_REGIONS.index(r) + 1] * w) for r in regions]

    def loss_fn(stage_local: bool):
        out = render(x, m if stage_local else None, p, scale=scale, active_global=gkeys, active_local=al if stage_local else {})
        ol = _lab(out)
        d = torch.sqrt(((ol - tl) ** 2).sum(-1) + 1e-4)[:, None]
        db = torch.sqrt(((_lab(blur(out, 3.0)) - tlb) ** 2).sum(-1) + 1e-4)[:, None]
        e = 0.5 * d + 0.5 * db
        loss = (e * w).sum() / wsum
        for rm in rw:  # balance small regions (eyes, lips) against large ones
            loss = loss + 0.25 * (e * rm).sum() / rm.sum().clamp(min=1)
        loss = loss + (greg * p.gv**2).sum() + (lreg * p.lv**2 * lmask).sum()
        return loss

    for stage_local, n, lr in ((False, steps // 2, 0.05), (bool(regions), steps - steps // 2, 0.03)):
        opt = torch.optim.Adam([p.gv, p.lv], lr=lr)
        for i in range(n):
            opt.zero_grad()
            loss = loss_fn(stage_local)
            loss.backward()
            p.gv.grad *= gmask
            p.lv.grad *= lmask if stage_local else 0
            opt.step()
            with torch.no_grad():
                p.gv.copy_(torch.maximum(torch.minimum(p.gv, ghi), glo))
                p.lv.copy_(torch.maximum(torch.minimum(p.lv, lhi), llo))

    recipe = p.to_recipe({"fit": {"steps": steps, "regions": list(regions)}}).pruned()
    with torch.no_grad():
        out = render(x, m, Params.from_recipe(recipe), scale=scale)
        de0 = delta_e76(srgb_to_lab(rs(orig)), srgb_to_lab(rs(target)))
        de1 = delta_e76(srgb_to_lab(out[0].permute(1, 2, 0).numpy()), srgb_to_lab(rs(target)))
    wn = w[0, 0].numpy()
    region_de = {}
    for r in regions:
        mm = m[0, LOCAL_REGIONS.index(r)].numpy() * wn
        region_de[r] = float((de1 * mm).sum() / max(mm.sum(), 1e-6))
    return FitResult(recipe, float((de0 * wn).sum() / wn.sum()), float((de1 * wn).sum() / wn.sum()), region_de)
