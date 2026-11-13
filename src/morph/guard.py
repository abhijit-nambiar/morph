"""Distortion guard for candidate targets (generated or hand-made).

A target is only used as *colour/tone evidence*; it must depict the same geometry as the original.
We check, at working resolution:

* geometric registration (is the candidate shifted / scaled / warped?  -> re-aligned if so),
* edge overlap (Canny edges of both, F1 with a small pixel tolerance),
* local structure correlation (fraction of pixels whose luminance detail still matches),
* semantic consistency (subject / hair / skin / clothes mask IoU and face count, when masks given).

The verdict is ``pass`` (use as-is), ``repairable`` (localized changes; the changed pixels are
masked out of the fit via ``confidence``) or ``reject``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .fit import _gray_u8, align, structure_confidence

THRESHOLDS = {
    "edge_f1_pass": 0.70,
    "edge_f1_min": 0.45,
    "structure_pass": 0.90,
    "structure_min": 0.70,
    "mask_iou_pass": 0.88,
    "mask_iou_min": 0.75,
}


@dataclass
class GuardReport:
    verdict: str
    edge_f1: float
    structure_ok: float
    alignment: str
    mask_iou: dict = field(default_factory=dict)
    faces: tuple[int, int] | None = None
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def edge_f1(a: np.ndarray, b: np.ndarray, tol: int | None = None) -> float:
    h, w = a.shape[:2]
    tol = tol or max(1, round(max(h, w) / 400))
    ea, eb = (cv2.Canny(cv2.GaussianBlur(_gray_u8(x), (0, 0), 1.0), 60, 140) > 0 for x in (a, b))
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    if ea.sum() < 50 and eb.sum() < 50:
        return 1.0
    prec = (eb & da).sum() / max(1, eb.sum())
    rec = (ea & db).sum() / max(1, ea.sum())
    return float(2 * prec * rec / max(prec + rec, 1e-6))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    return 1.0 if u < 30 else float((a & b).sum() / u)


def check(
    orig: np.ndarray,
    cand: np.ndarray,
    masks_orig: dict[str, np.ndarray] | None = None,
    masks_cand: dict[str, np.ndarray] | None = None,
    faces: tuple[int, int] | None = None,
) -> tuple[GuardReport, np.ndarray, np.ndarray]:
    """Returns (report, candidate aligned to ``orig``, per-pixel confidence in [0, 1])."""
    al = align(orig, cand)
    target, valid = al.target, al.valid
    conf = structure_confidence(orig, target) * valid
    reasons = []
    ef = edge_f1(orig, target)
    sv = float(conf[valid > 0].mean() > 0 and ((conf > 0.5) & (valid > 0)).sum() / max(1, (valid > 0).sum()))
    ious = {}
    if masks_orig is not None and masks_cand is not None:
        H, W = orig.shape[:2]
        for r in ("subject", "hair", "face_skin", "body_skin", "clothes"):
            if r in masks_orig and r in masks_cand:
                mc = masks_cand[r]
                if mc.shape != (H, W):
                    mc = cv2.resize(mc, (W, H), interpolation=cv2.INTER_LINEAR)
                if al.matrix is not None and al.method in ("homography", "ecc"):
                    mc = cv2.warpPerspective(cv2.resize(masks_cand[r], cand.shape[1::-1]), al.matrix, (W, H))
                if masks_orig[r].mean() > 0.003 or mc.mean() > 0.003:
                    ious[r] = round(_iou(masks_orig[r], mc), 4)
    if al.method != "identity":
        reasons.append(f"candidate was re-registered ({al.method})")
    verdict = "pass"

    def worse(v):
        nonlocal verdict
        order = ["pass", "repairable", "reject"]
        verdict = max(verdict, v, key=order.index)

    T = THRESHOLDS
    if ef < T["edge_f1_min"]:
        worse("reject"); reasons.append(f"edge overlap {ef:.2f} < {T['edge_f1_min']}")
    elif ef < T["edge_f1_pass"]:
        worse("repairable"); reasons.append(f"edge overlap {ef:.2f} below {T['edge_f1_pass']}")
    if sv < T["structure_min"]:
        worse("reject"); reasons.append(f"only {sv:.0%} of pixels keep their structure")
    elif sv < T["structure_pass"]:
        worse("repairable"); reasons.append(f"{1 - sv:.0%} of pixels changed structure (masked out of fit)")
    for r, v in ious.items():
        lim_min = T["mask_iou_min"] * (0.85 if r != "subject" else 1.0)
        if v < lim_min:
            worse("reject" if r == "subject" else "repairable"); reasons.append(f"{r} mask IoU {v:.2f}")
        elif v < T["mask_iou_pass"] * (0.9 if r != "subject" else 1.0):
            worse("repairable"); reasons.append(f"{r} mask IoU {v:.2f}")
    if faces is not None and faces[0] != faces[1]:
        worse("reject"); reasons.append(f"face count changed {faces[0]} -> {faces[1]}")
    if verdict == "pass":
        conf = np.maximum(conf, valid * 0.5)  # trust everything a little when nothing is wrong
    rep = GuardReport(verdict, round(ef, 4), round(sv, 4), al.method, ious, faces, reasons)
    return rep, target, conf.astype(np.float32)


def warp_test_candidate(img: np.ndarray, kind: str) -> np.ndarray:
    """Synthetic distortions used by tests and the self-check command."""
    h, w = img.shape[:2]
    if kind == "shift":
        M = np.float32([[1, 0, w * 0.03], [0, 1, h * 0.02]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if kind == "liquify":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy, r = w * 0.5, h * 0.4, min(h, w) * 0.25
        d = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        return cv2.remap(img, xx + 0.12 * r * d * np.sin(yy / 9), yy + 0.12 * r * d * np.cos(xx / 9), cv2.INTER_LINEAR)
    if kind == "inpaint":
        out = img.copy()
        y0, x0 = h // 3, w // 3
        out[y0 : y0 + h // 4, x0 : x0 + w // 4] = cv2.GaussianBlur(img, (0, 0), 25)[y0 : y0 + h // 4, x0 : x0 + w // 4]
        return out
    if kind == "replace":
        return np.ascontiguousarray(img[::-1])
    raise ValueError(kind)
