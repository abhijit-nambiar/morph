"""Edge-aware filters used for mask refinement (numpy / OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np


def box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Grey-guide guided filter (He et al.). guide: HxW, src: HxW or HxWxC float32."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    if src.ndim == 3:
        return np.stack([guided_filter(guide, src[..., c], r, eps) for c in range(src.shape[2])], -1)
    mean_i = box(guide, r)
    mean_p = box(src, r)
    cov_ip = box(guide * src, r) - mean_i * mean_p
    var_i = box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a, r) * guide + box(b, r)


def upsample_mask(mask_lo: np.ndarray, guide_hi: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Upsample a soft mask to the guide's resolution and snap it to the guide's edges."""
    h, w = guide_hi.shape[:2]
    up = cv2.resize(mask_lo.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    scale = max(1, round(max(h, w) / max(mask_lo.shape)))
    r = max(2, 2 * scale)
    g = guide_hi if guide_hi.ndim == 2 else cv2.cvtColor(guide_hi.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return np.clip(guided_filter(g, up, r, eps), 0.0, 1.0)


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
