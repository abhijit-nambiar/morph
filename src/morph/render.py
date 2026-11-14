"""Differentiable renderer for :class:`morph.recipe.Recipe`.

The same code renders low-resolution previews during fitting (with autograd) and full-resolution
exports. Every operation is a per-pixel tone/colour mapping, a mask-weighted blend, or an
edge-preserving detail filter, so geometry is always preserved.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .colorspace import LUMA, linear_to_srgb, srgb_to_linear
from .recipe import GLOBAL_KEYS, HSL_BANDS, LOCAL_KEYS, LOCAL_REGIONS, Recipe
from .segment import REGIONS

_LUMA = torch.tensor(LUMA, dtype=torch.float32)
# Opponent colour basis: Y (luma) plus two chroma axes orthogonal-ish to grey.
_M = torch.tensor(
    [list(LUMA), [1 / math.sqrt(2), -1 / math.sqrt(2), 0.0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]],
    dtype=torch.float32,
)
_MINV = torch.linalg.inv(_M)
_BAND_RGB = {
    "red": (1.0, 0.1, 0.1),
    "orange": (1.0, 0.55, 0.1),
    "yellow": (0.95, 0.95, 0.1),
    "green": (0.15, 0.85, 0.15),
    "aqua": (0.1, 0.85, 0.85),
    "blue": (0.1, 0.35, 1.0),
    "purple": (0.55, 0.1, 1.0),
    "magenta": (1.0, 0.1, 0.8),
}


def _band_angles() -> torch.Tensor:
    out = []
    for b in HSL_BANDS:
        rgb = torch.tensor(_BAND_RGB[b])
        o = _M @ rgb
        out.append(math.atan2(float(o[2]), float(o[1])))
    return torch.tensor(out)


_BAND_THETA = _band_angles()


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWx3 float numpy -> 1x3xHxW tensor."""
    return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)[None]


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x[0].permute(1, 2, 0).detach().cpu().numpy()


def masks_tensor(masks: dict[str, np.ndarray], names=REGIONS) -> torch.Tensor:
    shape = next(iter(masks.values())).shape
    zero = np.zeros(shape, np.float32)
    return torch.from_numpy(np.stack([masks.get(n, zero) for n in names], 0).astype(np.float32))[None]


def _luma(x: torch.Tensor) -> torch.Tensor:
    w = _LUMA.to(x).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def _gauss1d(sigma: float, device, dtype) -> torch.Tensor:
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur; large sigmas are computed on a decimated copy for speed."""
    if sigma < 0.3:
        return x
    if sigma > 6:
        f = int(sigma // 3)
        small = F.avg_pool2d(x, f, f, ceil_mode=True)
        out = blur(small, sigma / f)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
    k = _gauss1d(sigma, x.device, x.dtype)
    c = x.shape[1]
    r = k.numel() // 2
    xp = F.pad(x, (r, r, 0, 0), mode="reflect" if x.shape[-1] > r else "replicate")
    x = F.conv2d(xp, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    xp = F.pad(x, (0, 0, r, r), mode="reflect" if x.shape[-2] > r else "replicate")
    return F.conv2d(xp, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    if r < 1:
        return x
    k = 2 * r + 1
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    return F.avg_pool2d(xp, k, 1)


def _edge_preserving(x: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Self-guided filter on luma with chroma smoothing; used for noise reduction / skin smoothing."""
    y = _luma(x)
    mean_i = _box(y, r)
    var_i = _box(y * y, r) - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    y_s = _box(a, r) * y + _box(b, r)
    chroma = blur(x - y, max(1.0, r * 1.5))
    return y_s + chroma


def _bump(L: torch.Tensor, c: float, w: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((L - c) / w) ** 2)


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


class Params:
    """Tensor view of a recipe (global vector + [region, key] matrix)."""

    def __init__(self, g: torch.Tensor, l: torch.Tensor):
        self.gv = g
        self.lv = l
        self._gi = {k: i for i, k in enumerate(GLOBAL_KEYS)}
        self._li = {k: i for i, k in enumerate(LOCAL_KEYS)}

    @classmethod
    def from_recipe(cls, r: Recipe, requires_grad: bool = False) -> "Params":
        g = torch.tensor([r.g(k) for k in GLOBAL_KEYS], dtype=torch.float32)
        l = torch.tensor([[r.l(reg, k) for k in LOCAL_KEYS] for reg in LOCAL_REGIONS], dtype=torch.float32)
        if requires_grad:
            g.requires_grad_(True)
            l.requires_grad_(True)
        return cls(g, l)

    def to_recipe(self, meta: dict | None = None) -> Recipe:
        g = self.gv.detach().cpu().numpy()
        l = self.lv.detach().cpu().numpy()
        r = Recipe({k: float(g[i]) for i, k in enumerate(GLOBAL_KEYS)}, meta=meta or {})
        for ri, reg in enumerate(LOCAL_REGIONS):
            r.local[reg] = {k: float(l[ri, i]) for i, k in enumerate(LOCAL_KEYS)}
        return r.clamped().pruned(1e-4)

    def G(self, k: str) -> torch.Tensor:
        return self.gv[self._gi[k]]

    def L(self, ri: int, k: str) -> torch.Tensor:
        i = self._li.get(k)
        return self.lv[ri, i] if i is not None else self.lv.new_zeros(())


def _shoulder(lin: torch.Tensor, ev: torch.Tensor, knee: float = 0.6) -> torch.Tensor:
    """Roll highlights off toward white as exposure is pushed up, like a raw converter; identity at ev <= 0."""
    w = ev.clamp(0, 1)
    over = (lin - knee).clamp(min=0)
    rolled = knee + (1 - knee) * torch.tanh(over / (1 - knee))
    return lin + w * (torch.where(lin > knee, rolled, lin) - lin)


def _adjust(x: torch.Tensor, get, scale: float, active: set[str]) -> torch.Tensor:
    """Apply one block of sliders. ``get(key)`` returns a scalar tensor; ``scale`` = image long side."""
    # White balance + exposure in linear light.
    if active & {"temperature", "tint", "exposure"}:
        t, u, ev = get("temperature"), get("tint"), get("exposure")
        gains = torch.stack([torch.exp(0.30 * t + 0.10 * u), torch.exp(-0.20 * u), torch.exp(-0.30 * t + 0.10 * u)])
        gains = gains / (gains * _LUMA.to(x)).sum()
        lin = srgb_to_linear(x) * (gains.view(1, 3, 1, 1) * torch.pow(2.0, ev))
        x = linear_to_srgb(_shoulder(lin, ev))
    if active & {"contrast", "highlights", "shadows", "whites", "blacks", "dehaze", "curve_shadows", "curve_darks", "curve_lights", "curve_highlights"}:
        L = _luma(x).clamp(0, 1.2)
        Lc = L.clamp(0, 1)
        edge = (4 * Lc * (1 - Lc)).clamp(min=1e-6)  # floor keeps edge.sqrt() differentiable at pure black/white
        dh = get("dehaze")
        d = (get("contrast") + 0.35 * dh) * 0.55 * (Lc - 0.45) * edge
        d = d + get("highlights") * 0.30 * _bump(Lc, 0.78, 0.15) * edge.sqrt()
        d = d + get("shadows") * 0.30 * _bump(Lc, 0.22, 0.15) * edge.sqrt()
        d = d + get("whites") * 0.20 * _bump(Lc, 0.94, 0.08)
        d = d + (get("blacks") - 0.25 * dh) * 0.15 * _bump(Lc, 0.04, 0.07)
        d = d + get("curve_shadows") * 0.15 * _bump(Lc, 0.12, 0.08) * edge.sqrt()
        d = d + get("curve_darks") * 0.15 * _bump(Lc, 0.36, 0.11) * edge.sqrt()
        d = d + get("curve_lights") * 0.15 * _bump(Lc, 0.62, 0.11) * edge.sqrt()
        d = d + get("curve_highlights") * 0.15 * _bump(Lc, 0.86, 0.08) * edge.sqrt()
        x = x + d
    if active & {"clarity", "texture", "noise_reduction", "dehaze"}:
        L = _luma(x)
        cl = get("clarity") + 0.3 * get("dehaze")
        if "clarity" in active or "dehaze" in active:
            base = blur(L, 0.02 * scale)
            mid = (4 * L.clamp(0, 1) * (1 - L.clamp(0, 1))).clamp(min=0)
            x = x + cl * 0.8 * (L - base) * mid
        if "texture" in active:
            tx = get("texture")
            sm = blur(x, max(0.6, 0.0012 * scale))
            x = x + tx * (x - sm)
        if "noise_reduction" in active:
            r = max(1, int(round(0.0008 * scale)))
            x = x + get("noise_reduction") * (_edge_preserving(x, r, 2e-3) - x)
    if active & {"saturation", "vibrance", "hue", "dehaze"} or any(k.startswith(("hsl_", "grade_")) for k in active):
        m = _M.to(x)
        o = torch.einsum("ij,bjhw->bihw", m, x)
        Y, o1, o2 = o[:, :1], o[:, 1:2], o[:, 2:3]
        rho = torch.sqrt(o1 * o1 + o2 * o2 + 1e-8)
        grey = (o1.abs() + o2.abs()) < 1e-6  # atan2(0, 0) has NaN gradients
        theta = torch.atan2(torch.where(grey, torch.zeros_like(o2), o2), torch.where(grey, torch.ones_like(o1), o1))
        sat = 1 + get("saturation") + 0.25 * get("dehaze")
        if "vibrance" in active:
            sat = sat * (1 + get("vibrance") * (1 - (rho * 4).clamp(0, 1)))
        dtheta = get("hue") * (math.pi / 6)
        dlum = torch.zeros_like(Y)
        if any(k.startswith("hsl_") for k in active):
            bt = _BAND_THETA.to(x).view(1, -1, 1, 1)
            w = torch.exp(4.0 * (torch.cos(theta - bt) - 1))
            w = w / (w.sum(1, keepdim=True) + 1e-6)
            hs = torch.stack([get(f"hsl_{b}_hue") for b in HSL_BANDS]).view(1, -1, 1, 1)
            ss = torch.stack([get(f"hsl_{b}_sat") for b in HSL_BANDS]).view(1, -1, 1, 1)
            ls = torch.stack([get(f"hsl_{b}_lum") for b in HSL_BANDS]).view(1, -1, 1, 1)
            dtheta = dtheta + (w * hs).sum(1, keepdim=True) * (math.pi / 8)
            sat = sat * (1 + (w * ss).sum(1, keepdim=True))
            dlum = (w * ls).sum(1, keepdim=True) * 0.25 * (rho * 5).clamp(0, 1)
        rho = rho * sat.clamp(min=0)
        theta = theta + dtheta
        o1, o2 = rho * torch.cos(theta), rho * torch.sin(theta)
        Y = Y + dlum
        if any(k.startswith("grade_") for k in active):
            Lc = Y.clamp(0, 1)
            zones = (1 - _smoothstep(0.0, 0.55, Lc), _bump(Lc, 0.5, 0.2), _smoothstep(0.45, 1.0, Lc))
            for z, wz in zip(("shadows", "midtones", "highlights"), zones):
                o1 = o1 + wz * get(f"grade_{z}_x") * 0.08
                o2 = o2 + wz * get(f"grade_{z}_y") * 0.08
        x = torch.einsum("ij,bjhw->bihw", _MINV.to(x), torch.cat([Y, o1, o2], 1))
    return x


def _soft_clip(x: torch.Tensor) -> torch.Tensor:
    """Gamut-map out-of-range pixels toward their luma (hue-preserving); identity in gamut."""
    Y = _luma(x).clamp(0.0, 1.0)
    mx = x.amax(1, keepdim=True)
    mn = x.amin(1, keepdim=True)
    t_hi = torch.where(mx > 1, (1 - Y) / (mx - Y).clamp(min=1e-6), torch.ones_like(Y))
    t_lo = torch.where(mn < 0, Y / (Y - mn).clamp(min=1e-6), torch.ones_like(Y))
    t = torch.minimum(t_hi, t_lo).clamp(0, 1)
    return (Y + t * (x - Y)).clamp(0.0, 1.0)


def _mask_box(m: torch.Tensor, margin: int, eps: float = 1e-3):
    """Bounding box (with margin for spatial filters) of a 1x1xHxW mask, or None if empty."""
    with torch.no_grad():
        on = m[0, 0] > eps
        rows, cols = on.any(1).nonzero(), on.any(0).nonzero()
    if rows.numel() == 0:
        return None
    H, W = m.shape[-2:]
    return (max(0, int(rows[0]) - margin), min(H, int(rows[-1]) + 1 + margin), max(0, int(cols[0]) - margin), min(W, int(cols[-1]) + 1 + margin))


def _active(values: dict[str, float], eps: float = 1e-4) -> set[str]:
    return {k for k, v in values.items() if abs(v) > eps}


def render(
    img: torch.Tensor,
    masks: torch.Tensor | None,
    params: Params,
    *,
    scale: float | None = None,
    active_global: set[str] | None = None,
    active_local: dict[int, set[str]] | None = None,
    seed: int = 0,
    gamut_map: bool = True,
) -> torch.Tensor:
    """Render ``img`` (1x3xHxW sRGB) with ``params``; ``masks`` is 1xRxHxW in REGIONS order.

    ``gamut_map=False`` returns the unclipped result (no gamut mapping, no grain) so callers can see overshoot."""
    scale = float(scale or max(img.shape[-2:]))
    x = img
    ag = set(GLOBAL_KEYS) if active_global is None else active_global
    if ag:
        x = _adjust(x, params.G, scale, ag)
    if masks is not None:
        for ri, reg in enumerate(LOCAL_REGIONS):
            keys = set(LOCAL_KEYS) if active_local is None else active_local.get(ri, set())
            if not keys:
                continue
            m = masks[:, ri : ri + 1]
            box = _mask_box(m, int(0.07 * scale) + 4)
            if box is None:
                continue
            y0, y1, x0, x1 = box
            xc, mc = x[..., y0:y1, x0:x1], m[..., y0:y1, x0:x1]
            yc = _adjust(xc, lambda k, ri=ri: params.L(ri, k), scale, keys)
            if x.shape[-2:] == yc.shape[-2:]:
                x = x + mc * (yc - xc)
            else:
                x = torch.cat([x[..., :y0, :], torch.cat([x[..., y0:y1, :x0], xc + mc * (yc - xc), x[..., y0:y1, x1:]], -1), x[..., y1:, :]], -2)
    if "vignette" in ag:
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        rr = (xx * xx + yy * yy) / 2
        x = x * (1 + params.G("vignette") * 0.6 * rr * rr)
    if "sharpen" in ag:
        L = _luma(x)
        x = x + params.G("sharpen") * 0.8 * (L - blur(L, max(0.7, 0.0004 * scale)))
    if not gamut_map:
        return x
    x = _soft_clip(x)
    if "grain" in ag and float(params.G("grain").detach()) > 1e-3:
        x = apply_grain(x, params.G("grain"), seed, scale)
    return x


def apply_grain(x: torch.Tensor, amount, seed: int, scale: float) -> torch.Tensor:
    """Film grain, the last step of :func:`render`: seeded per-pixel noise, never structural."""
    gen = torch.Generator().manual_seed(seed)
    h, w = x.shape[-2:]
    size = max(1, int(round(0.0005 * scale)))
    n = torch.randn((1, 1, (h + size - 1) // size, (w + size - 1) // size), generator=gen)
    n = F.interpolate(n, size=(h, w), mode="bilinear", align_corners=False)
    L = _luma(x).clamp(0, 1)
    return (x + amount * 0.05 * n * (4 * L * (1 - L)).clamp(min=0.15)).clamp(0, 1)


def add_grain(img: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Numpy :func:`apply_grain`; ``add_grain(render_recipe(img, m, r without grain), g, seed)`` equals the full render."""
    if amount <= 1e-3:
        return img
    with torch.no_grad():
        return to_numpy(apply_grain(to_tensor(img), torch.tensor(float(amount)), seed, float(max(img.shape[:2]))))


def render_recipe(img: np.ndarray, masks: dict[str, np.ndarray] | None, recipe: Recipe, seed: int = 0) -> np.ndarray:
    """Numpy convenience wrapper that only evaluates the sliders that are actually set."""
    r = recipe.pruned(1e-4)
    p = Params.from_recipe(r)
    ag = _active(r.global_)
    al = {LOCAL_REGIONS.index(reg): _active(d) for reg, d in r.local.items() if reg in LOCAL_REGIONS}
    with torch.no_grad():
        mt = masks_tensor(masks) if masks is not None and al else None
        out = render(to_tensor(img), mt, p, active_global=ag, active_local=al, seed=seed)
    return to_numpy(out)
