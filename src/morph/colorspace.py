"""Color conversions shared by numpy and torch code paths (sRGB, linear, CIELAB)."""

from __future__ import annotations

import numpy as np
import torch

_M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float32,
)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(x):
    if isinstance(x, torch.Tensor):
        x = x.clamp(min=0.0)
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    if isinstance(x, torch.Tensor):
        x = x.clamp(min=0.0)
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 1e-8) ** (1 / 2.4) - 0.055).astype(np.float32)


def _f(t):
    d = 6.0 / 29.0
    if isinstance(t, torch.Tensor):
        return torch.where(t > d**3, t.clamp(min=1e-8) ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)
    return np.where(t > d**3, np.cbrt(np.maximum(t, 1e-8)), t / (3 * d * d) + 4.0 / 29.0)


def _finv(t):
    d = 6.0 / 29.0
    if isinstance(t, torch.Tensor):
        return torch.where(t > d, t**3, 3 * d * d * (t - 4.0 / 29.0))
    return np.where(t > d, t**3, 3 * d * d * (t - 4.0 / 29.0))


def srgb_to_lab(x):
    """sRGB (..., 3) in [0,1] -> CIELAB (..., 3) with L in [0,100]."""
    lin = srgb_to_linear(x)
    if isinstance(x, torch.Tensor):
        m = torch.as_tensor(_M_RGB2XYZ, dtype=x.dtype, device=x.device)
        w = torch.as_tensor(_WHITE, dtype=x.dtype, device=x.device)
        xyz = lin @ m.T / w
        f = _f(xyz)
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        return torch.stack([L, a, b], -1)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = _f(xyz)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1).astype(
        np.float32
    )


def lab_to_srgb(lab):
    if isinstance(lab, torch.Tensor):
        fy = (lab[..., 0] + 16) / 116
        fx = fy + lab[..., 1] / 500
        fz = fy - lab[..., 2] / 200
        w = torch.as_tensor(_WHITE, dtype=lab.dtype, device=lab.device)
        xyz = torch.stack([_finv(fx), _finv(fy), _finv(fz)], -1) * w
        minv = torch.as_tensor(np.linalg.inv(_M_RGB2XYZ), dtype=lab.dtype, device=lab.device)
        return linear_to_srgb(xyz @ minv.T)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xyz = np.stack([_finv(fx), _finv(fy), _finv(fz)], -1) * _WHITE
    return linear_to_srgb(xyz @ np.linalg.inv(_M_RGB2XYZ).T.astype(np.float32))


def luma(x):
    if isinstance(x, torch.Tensor):
        return x[..., 0] * LUMA[0] + x[..., 1] * LUMA[1] + x[..., 2] * LUMA[2]
    return (x[..., 0] * LUMA[0] + x[..., 1] * LUMA[1] + x[..., 2] * LUMA[2]).astype(np.float32)


def delta_e76(a_lab, b_lab):
    return np.linalg.norm(a_lab - b_lab, axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (numpy, vectorised)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lpm = (L1 + L2) / 2
    Cpm = (C1p + C2p) / 2
    hsum = h1p + h2p
    hpm = np.where(
        C1p * C2p == 0,
        hsum,
        np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hpm - 30))
        + 0.24 * np.cos(np.radians(2 * hpm))
        + 0.32 * np.cos(np.radians(3 * hpm + 6))
        - 0.20 * np.cos(np.radians(4 * hpm - 63))
    )
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm**7 / (Cpm**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))
