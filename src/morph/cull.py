"""Auto-cull a camera dump: score every frame, group bursts, keep the best, build the base album.

Per frame (cached, so re-scoring after changing thresholds is instant):

* **focus** where it matters: the eyes of the main faces when people are present, else the
  people, else the sharpest part of the frame (so shallow depth of field is not penalised).
  Expressed as an estimated blur radius in pixels at a fixed ``MEASURE_SIDE`` view, which makes
  the thresholds content- and contrast-independent.
* **motion blur**: how much more blurred the focus area is along one direction than across it.
* **exposure**: subject/face key brightness in EV relative to a pleasant mid-tone, plus clipping.
* **closed eyes** per face, from the face-parsing model's eye masks.
* **burst / near-duplicate** groups from capture time + a small appearance descriptor.

``decide`` turns scores + :class:`CullSettings` into keep/reject decisions with human-readable
reasons; ``build_album`` links or copies the keepers into a folder for ``morph init``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from .imageio import is_raw, list_images, load_preview, raw_pairs, read_info

SCORE_VERSION = "cull-v3"
MEASURE_SIDE = 2560  # focus is judged as if viewing the photo at this long side (a large screen / A4 print)
DETECT_SIDE = 512
MAX_FACES = 6
MAX_PARSED_FACES = 4  # faces run through the face parser (eyes), largest first
FACE_PARSE_SIDE = 384  # face parser input size
EYE_MIN_FACE_PX = 80  # face width (at MEASURE_SIDE) needed to judge eye state
SHUT_MIN_FACE_PX = 300  # ...and to infer shut eyes from brows without visible eyes
FACE_OUT_SIDE = 320  # face parser output size (eye geometry is measured at this resolution)
REPORT_NAME = "cull.json"
KEY_TARGET = 0.08  # log-mean linear luminance of a well-exposed, display-ready photo (median of the 88 real originals)
_MANIFEST = ".morph-cull-album.json"


# -- measurements ----------------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img, dtype=np.float32), cv2.COLOR_RGB2GRAY)


def _energy_ratio_sigma(e0: float, e1: float, s: float) -> float:
    """Blur radius from gradient energy before/after an extra Gaussian blur of ``s`` px.

    For an edge blurred by sigma, total squared-gradient energy scales with 1/sigma, so
    ``e0 / e1 = sqrt(sigma^2 + s^2) / sigma``: contrast cancels out.
    """
    r = e0 / max(e1, 1e-12)
    if r <= 1.0001:
        return 50.0
    return float(min(50.0, s / math.sqrt(r * r - 1.0)))


class _Grad:
    """Gradient energy maps of a grey image, before and after a reference blur."""

    S = 1.5

    def __init__(self, gray: np.ndarray):
        g = cv2.GaussianBlur(gray, (0, 0), 0.5)  # tames sensor noise / JPEG blocking
        b = cv2.GaussianBlur(g, (0, 0), self.S)
        self.gx0, self.gy0 = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        self.gx1, self.gy1 = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        self.e0 = self.gx0**2 + self.gy0**2
        self.e1 = self.gx1**2 + self.gy1**2
        self._g = g

    def edge_mask(self, box=None, frac: float = 0.12) -> np.ndarray:
        """Pixels near the strongest edges (top ``frac`` of blurred gradient), dilated to cover edge profiles."""
        e1 = self.e1 if box is None else self.e1[box[1] : box[3], box[0] : box[2]]
        if e1.size < 64:
            return np.zeros_like(e1, bool)
        thr = np.quantile(e1, 1 - frac)
        m = (e1 >= max(thr, 1e-7)).astype(np.uint8)
        return cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)

    def sigma(self, box=None) -> float | None:
        sl = (slice(None), slice(None)) if box is None else (slice(box[1], box[3]), slice(box[0], box[2]))
        e0, e1 = self.e0[sl], self.e1[sl]
        m = self.edge_mask(box)
        if m.sum() < 30 or float(e1[m].mean()) < 1e-5:  # featureless: no evidence either way
            return None
        return _energy_ratio_sigma(float(e0[m].sum()), float(e1[m].sum()), self.S)

    def tile_sigmas(self, size: int, weight: np.ndarray | None = None, min_weight: float = 0.3) -> np.ndarray:
        h, w = self.e0.shape
        out = []
        for y in range(0, h - size // 2, size):
            for x in range(0, w - size // 2, size):
                box = (x, y, min(w, x + size), min(h, y + size))
                if weight is not None and weight[box[1] : box[3], box[0] : box[2]].mean() < min_weight:
                    continue
                s = self.sigma(box)
                if s is not None:
                    out.append(s)
        return np.array(out, np.float32)

    def motion(self, box=None) -> tuple[float, float]:
        """(anisotropy 0..1, direction in degrees) of blur inside ``box``.

        Per direction, compares directional-derivative energy before/after a 1-D blur along that
        direction; defocus blurs every direction equally, shake/subject motion mostly one.
        """
        sl = (slice(None), slice(None)) if box is None else (slice(box[1], box[3]), slice(box[0], box[2]))
        g = self._g[sl]
        if min(g.shape) < 24:
            return 0.0, 0.0
        gx0, gy0 = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        m = self.edge_mask(box)
        if m.sum() < 30:
            return 0.0, 0.0
        sig = []
        angles = (0, 45, 90, 135)
        for a in angles:
            k = _line_kernel(a, self.S)
            b = cv2.filter2D(g, cv2.CV_32F, k)
            c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
            d0 = c * gx0 + s * gy0
            d1 = c * cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) + s * cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
            sig.append(_energy_ratio_sigma(float((d0[m] ** 2).sum()), float((d1[m] ** 2).sum()), self.S))
        sig = np.array(sig)
        hi, lo = float(sig.max()), float(sig.min())
        return float((hi - lo) / (hi + lo + 0.5)), float(angles[int(sig.argmax())])


def _line_kernel(angle_deg: float, s: float) -> np.ndarray:
    r = int(math.ceil(3 * s))
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    c, sn = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for t in np.linspace(-r, r, 8 * r + 1):
        x, y = r + t * c, r + t * sn
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        wgt = math.exp(-0.5 * (t / s) ** 2)
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx <= 2 * r and 0 <= y0 + dy <= 2 * r:
                k[y0 + dy, x0 + dx] += wgt * ww
    return k / k.sum()


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def exposure_stats(small: np.ndarray, weight: np.ndarray | None = None) -> dict:
    """Key brightness (EV vs. a pleasant mid-tone) and clipping, subject-weighted when given."""
    lin = _srgb_to_linear(np.clip(small, 0, 1)) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    w = np.ones_like(lin) if weight is None or weight.sum() < 50 else 0.35 + weight
    key = float(np.exp((np.log(lin + 1e-3) * w).sum() / w.sum()))
    hi = (small.max(-1) >= 0.995).astype(np.float32)
    lo = (lin <= 0.002).astype(np.float32)
    out = {"key_ev": round(math.log2(key / KEY_TARGET), 3), "clip_hi": round(float(hi.mean()), 4), "clip_lo": round(float(lo.mean()), 4)}
    if weight is not None and weight.sum() >= 50:
        out["subject_clip_hi"] = round(float((hi * weight).sum() / weight.sum()), 4)
    else:
        out["subject_clip_hi"] = out["clip_hi"]
    return out


def descriptor(small: np.ndarray) -> np.ndarray:
    """Appearance vector for near-duplicate detection: a blurred, normalised tiny grey image plus coarse colour."""
    g = cv2.resize(_gray(small), (32, 24), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 1.0).ravel()
    g = (g - g.mean()) / (g.std() + 1e-4)
    col = cv2.resize(small, (4, 3), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([g / math.sqrt(g.size), col]).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """0..1: structural correlation of the tiny grey images, discounted by colour difference."""
    n = 32 * 24
    corr = float(np.dot(a[:n], b[:n]))
    dcol = float(np.abs(a[n:] - b[n:]).mean())
    return max(0.0, corr) * math.exp(-dcol / 0.12)


# -- people ----------------------------------------------------------------------------------------
class PeopleDetector:
    """Person mask, faces and eye state from the segmentation models in :mod:`morph.segment`.

    Runs the human parser at ``DETECT_SIDE`` and the face parser on up to ``MAX_FACES`` face
    crops, which is far cheaper than full album analysis.
    """

    def __init__(self, segmenter=None):
        from .segment import Segmenter

        self.seg = segmenter or Segmenter()

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        from .segment import FACE_MODEL, HUMAN_MODEL, _components, _group_probs, _HUMAN_GROUPS

        H, W = img.shape[:2]
        s = DETECT_SIDE / max(H, W)
        small = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        u8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
        probs, labels = self.seg._probs(HUMAN_MODEL, u8)
        g = _group_probs(probs, labels, _HUMAN_GROUPS)
        person = np.clip(1 - g["background"], 0, 1).astype(np.float32)
        comps = sorted(_components(g["face"], 0.5, min_frac=0.0002, scale=1.0), key=lambda c: -c["area_frac"])
        full_u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        faces = []
        parsed = 0
        for c in comps[:MAX_FACES]:
            x0, y0, x1, y1 = (v / s for v in c["box"])
            fw, fh = x1 - x0, y1 - y0
            if min(fw, fh) < 16:
                continue
            if parsed >= MAX_PARSED_FACES or c["area_frac"] < 0.25 * comps[0]["area_frac"] or fw < 40:
                # background faces: too small to judge eyes; focus is still measured on the face box
                faces.append({"box": [round(x0), round(y0), round(x1), round(y1)], "area_frac": round(fw * fh / (W * H), 5), "eyes": []})
                continue
            parsed += 1
            side = max(fw, fh) * 1.6
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            bx1, by1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            crop = full_u8[by0:by1, bx0:bx1]
            k = min(1.0, FACE_OUT_SIDE / max(crop.shape[:2]))
            fp, flab = self.seg._probs(FACE_MODEL, crop, out_hw=(max(1, round(crop.shape[0] * k)), max(1, round(crop.shape[1] * k))), resize_to=FACE_PARSE_SIDE)
            lab = {str(v).lower(): int(k) for k, v in flab.items()}
            face = {"box": [round(x0), round(y0), round(x1), round(y1)], "area_frac": round(fw * fh / (W * H), 5), "eyes": []}
            skin = fp[[lab["skin"], lab["nose"]]].sum(0) if "skin" in lab else None
            ch, cw = fp.shape[1:]
            ks = crop.shape[0] / ch
            for e in ("l_eye", "r_eye"):
                if e not in lab:
                    continue
                m = fp[lab[e]] > 0.35
                if m.sum() < 6:
                    continue
                ys, xs = np.nonzero(m)
                ew, eh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
                face["eyes"].append({
                    "box": [round(bx0 + xs.min() * ks), round(by0 + ys.min() * ks), round(bx0 + (xs.max() + 1) * ks), round(by0 + (ys.max() + 1) * ks)],
                    "aspect": round(float(eh / ew), 3),
                    "area": round(float(m.sum() / (ch * cw)), 5),
                })
            face["brows"] = sum(int((fp[lab[b]] > 0.4).sum() >= 6) for b in ("l_brow", "r_brow") if b in lab)
            if not face["eyes"] and "eye_g" in lab and (fp[lab["eye_g"]] > 0.5).sum() > 20:
                ys, xs = np.nonzero(fp[lab["eye_g"]] > 0.5)  # glasses hide the eyes: still a good focus target
                face["glasses"] = [round(bx0 + xs.min() * ks), round(by0 + ys.min() * ks), round(bx0 + (xs.max() + 1) * ks), round(by0 + (ys.max() + 1) * ks)]
            face["skin_frac"] = round(float((skin > 0.5).mean()), 4) if skin is not None else None
            faces.append(face)
        person_full = cv2.resize(person, (W, H), interpolation=cv2.INTER_LINEAR)
        return person_full, faces


def eye_openness(face: dict) -> float | None:
    """0 (closed) .. 1 (clearly open) for a face, None when no eyes are visible (profile, sunglasses, back of head)."""
    eyes = face.get("eyes") or []
    width = face["box"][2] - face["box"][0]
    if width < EYE_MIN_FACE_PX:
        return None  # too small to tell a blink from a squint
    if not eyes:
        # A large frontal face (brows visible, no glasses) whose eyes the parser cannot find: lids are shut.
        return 0.05 if width >= SHUT_MIN_FACE_PX and face.get("brows", 0) >= 1 and not face.get("glasses") else None
    best = max(e["aspect"] for e in eyes)
    return float(np.clip((best - 0.10) / (0.25 - 0.10), 0, 1))


# -- per-frame score -------------------------------------------------------------------------------
@dataclass
class FrameScore:
    name: str
    path: str
    scored_from: str  # file the pixels came from (RAW embedded preview, or the paired camera JPEG)
    captured: str | None
    width: int
    height: int
    focus_region: str  # "eyes" | "faces" | "subject" | "frame"
    focus_blur_px: float | None  # estimated blur radius at MEASURE_SIDE in the focus region (lower = sharper)
    frame_blur_px: float | None  # sharpest part of the whole frame
    subject_blur_px: float | None
    eye_blur_px: float | None
    motion: float  # 0 (none) .. 1 (strongly directional blur)
    motion_angle: float
    key_ev: float  # subject-weighted brightness relative to a pleasant mid-tone; 0 = well exposed
    clip_hi: float
    clip_lo: float
    subject_clip_hi: float
    faces: list = field(default_factory=list)
    n_faces: int = 0
    eyes_closed: int = 0  # faces whose eyes look closed
    eyes_open: int = 0
    raw: bool = False
    iso: float | None = None
    shutter: float | None = None
    burst_id: int = -1
    burst_rank: int = 1  # 1 = best frame of its burst
    burst_size: int = 1
    focus: float = 0.0  # 0..1 sub-scores (1 = perfect) under the default settings
    exposure: float = 0.0
    stability: float = 0.0  # 1 - motion blur
    eyes: float = 1.0
    quality: float = 0.0  # 0..1 overall (product of the sub-scores)
    reasons: list = field(default_factory=list)  # default-settings verdict, human readable
    keep: bool | None = None  # default-settings verdict
    sub_scores: dict = field(default_factory=dict)
    preview: str | None = None  # small JPEG thumbnail in the cache, for UIs
    descriptor: list | None = None

    def to_json(self, descriptor: bool = False) -> dict:
        d = asdict(self)
        if not descriptor:
            d.pop("descriptor")
        return d

    @classmethod
    def from_json(cls, d: dict) -> "FrameScore":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _score_pixels(img: np.ndarray, detector) -> dict:
    H, W = img.shape[:2]
    gray = _gray(img)
    G = _Grad(gray)
    tile = max(64, round(max(H, W) / 20))
    person, faces = detector(img) if detector is not None else (None, [])
    tiles = G.tile_sigmas(tile)
    frame_blur = float(np.percentile(tiles, 10)) if len(tiles) else None
    subject_blur = None
    if person is not None and person.mean() > 0.01:
        st = G.tile_sigmas(tile, person, 0.35)
        if len(st):
            subject_blur = float(np.percentile(st, 25))
    # Faces: the main ones (at least half the area of the largest) decide focus; eyes when big enough to judge.
    main = [f for f in faces if faces and f["area_frac"] >= 0.5 * faces[0]["area_frac"]][:4]
    eye_vals, face_vals, motion_boxes = [], [], []
    for f in faces:
        x0, y0, x1, y1 = f["box"]
        f["blur_px"] = G.sigma((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
        eb = [e["box"] for e in f.get("eyes", [])] or ([f["glasses"]] if f.get("glasses") else [])
        f["eye_blur_px"] = None
        if eb and (x1 - x0) >= 48:
            ex0, ey0 = min(b[0] for b in eb), min(b[1] for b in eb)
            ex1, ey1 = max(b[2] for b in eb), max(b[3] for b in eb)
            pad = max(6, round(0.12 * (x1 - x0)))  # include lashes, lids and brows
            f["eye_blur_px"] = G.sigma((max(0, ex0 - pad), max(0, ey0 - pad), min(W, ex1 + pad), min(H, ey1 + pad)))
        f["openness"] = eye_openness(f)
        if f in main:
            if f["eye_blur_px"] is not None:
                eye_vals.append(f["eye_blur_px"])
            if f["blur_px"] is not None:
                face_vals.append(f["blur_px"])
            motion_boxes.append((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    eye_blur = float(np.median(eye_vals)) if eye_vals else None
    if eye_blur is not None and face_vals:  # eyes are tiny: guard against a lucky crop with the face's own measure
        region, focus = "eyes", float(max(eye_blur, 0.8 * np.median(face_vals)))
    elif face_vals:
        region, focus = "faces", float(np.median(face_vals))
    elif subject_blur is not None:
        region, focus = "subject", subject_blur
    else:
        region, focus = "frame", frame_blur
    # Motion: on the faces, or on the sharpest tiles of the subject / frame.
    if motion_boxes:
        mot = [G.motion(b) for b in motion_boxes]
        motion, angle = max(mot)
    else:
        motion, angle = G.motion(None)
    s = DETECT_SIDE / max(H, W)
    small = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    wmask = None
    if person is not None:
        wmask = cv2.resize(person, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        fm = np.zeros_like(wmask)
        for f in main:
            x0, y0, x1, y1 = (round(v * s) for v in f["box"])
            fm[y0:y1, x0:x1] = 2.0
        wmask = wmask + fm
    ex = exposure_stats(small, wmask)
    judged = [f for f in faces if f["openness"] is not None and (f["box"][2] - f["box"][0]) >= 40]
    return {
        "focus_region": region, "focus_blur_px": focus, "frame_blur_px": frame_blur, "subject_blur_px": subject_blur,
        "eye_blur_px": eye_blur, "motion": round(motion, 4), "motion_angle": angle, **ex,
        "faces": faces, "n_faces": len(faces),
        "eyes_closed": sum(f["openness"] < 0.25 for f in judged), "eyes_open": sum(f["openness"] >= 0.25 for f in judged),
        "descriptor": descriptor(small).round(4).tolist(), "_small": small,
    }


def _signature(p: Path) -> str:
    st = p.stat()
    return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"


def _round(d):
    if isinstance(d, float):
        return round(d, 4)
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_round(v) for v in d]
    return d


def _relocate(rec: dict, f: Path, folder: Path, cache: Path) -> dict:
    """A cached record with its paths pointed at the current files (the dump or cache may have moved)."""
    rec = dict(rec)
    rec["path"] = str(f.resolve())
    src = folder / Path(rec.get("scored_from") or f).name
    rec["scored_from"] = str(src.resolve()) if src.exists() else rec["path"]
    if rec.get("preview"):
        prev = cache / "previews" / Path(rec["preview"]).name
        rec["preview"] = str(prev.resolve()) if prev.exists() else None
    return rec


def score_folder(folder: str | Path, cache: str | Path | None = None, progress=None, people: bool = True,
                 detector=None, pairs: str = "raw", workers: int | None = None) -> list[FrameScore]:
    """Score every frame in ``folder`` (RAW and/or JPEG). Results are cached in ``cache``
    (default ``<folder>/.morph-cull``) keyed by file name, size and mtime.

    ``people=False`` skips the neural person/face models (much faster; focus then uses the
    sharpest region of the frame). ``detector`` overrides the people detector (tests).
    ``progress(i, n, name)`` is called after each frame. Returned in capture order with burst
    groups and default-settings quality filled in.
    """
    folder = Path(folder)
    cache = Path(cache) if cache else folder / ".morph-cull"
    (cache / "previews").mkdir(parents=True, exist_ok=True)
    idx_path = cache / "scores.json"
    try:
        prior = json.loads(idx_path.read_text())
        prior = prior["frames"] if prior.get("version") == SCORE_VERSION and prior.get("people") == people else {}
    except (OSError, ValueError, KeyError):
        prior = {}
    files = list_images(folder, pairs=pairs)
    sidecars = raw_pairs(folder) if pairs != "jpeg" else {}
    if detector is None and people:
        detector = PeopleDetector()
    todo = [f for f in files if _signature(f) not in prior]

    def load(f: Path):
        src = sidecars.get(f.stem, f) if is_raw(f) else f
        info = read_info(f, with_hash=False)
        return f, src, info, load_preview(src, MEASURE_SIDE)

    out: dict[str, dict] = {}
    n, done = len(files), 0
    for f in files:
        sig = _signature(f)
        if sig in prior:
            out[sig] = _relocate(prior[sig], f, folder, cache)
            done += 1
            if progress:
                progress(done, n, f.name)
    workers = workers or min(4, os.cpu_count() or 2)
    with ThreadPoolExecutor(workers) as pool:  # decode ahead while the models run, at most 2*workers frames in memory
        pending, window = iter(todo), deque()
        for f in islice(pending, 2 * workers):
            window.append(pool.submit(load, f))
        while window:
            f, src, info, img = window.popleft().result()
            nxt = next(pending, None)
            if nxt is not None:
                window.append(pool.submit(load, nxt))
            r = _score_pixels(img, detector)
            del img
            small = r.pop("_small")
            sig = _signature(f)
            prev = cache / "previews" / (hashlib.sha1(sig.encode()).hexdigest()[:16] + ".jpg")
            cv2.imwrite(str(prev), cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            out[sig] = _round({
                "name": f.name, "path": str(f.resolve()), "scored_from": str(src.resolve()),
                "captured": info.captured.isoformat() if info.captured else None,
                "width": info.width, "height": info.height, "raw": is_raw(f), "iso": info.iso, "shutter": info.exposure_time,
                "preview": str(prev.resolve()), **r,
            })
            done += 1
            if progress:
                progress(done, n, f.name)
            if done % 25 == 0:
                idx_path.write_text(json.dumps({"version": SCORE_VERSION, "people": people, "frames": {**prior, **out}}))
    idx_path.write_text(json.dumps({"version": SCORE_VERSION, "people": people, "frames": {**prior, **out}}))
    scores = [FrameScore.from_json(out[_signature(f)]) for f in files]
    scores.sort(key=lambda s: (s.captured or "", s.name))
    settings = CullSettings()
    group_bursts(scores, settings)
    verdicts = {d.name: d for d in decide(scores, settings)}
    for s in scores:
        d = verdicts[s.name]
        s.quality, sub = quality(s, settings)
        s.focus, s.exposure, s.stability, s.eyes = sub["focus"], sub["exposure"], sub["stability"], sub["eyes"]
        s.burst_id, s.burst_rank, s.burst_size, s.keep, s.reasons = d.burst_id, d.burst_rank, d.burst_size, d.keep, d.reasons
    return scores


# -- decisions -------------------------------------------------------------------------------------
@dataclass
class CullSettings:
    """User-tunable culling thresholds. Defaults aim to keep everything a photographer would
    consider for an album and drop only clear failures and redundant burst frames."""

    focus_strictness: float = 0.5  # 0 lenient .. 1 strict; maps to the max blur allowed in the focus region
    max_motion: float = 0.2  # reject when blur is this directional (camera shake / subject motion)
    exposure_tolerance_ev: float = 1.5  # how far subject brightness may sit from a good mid-tone
    raw_latitude_ev: float = 1.0  # extra under-exposure tolerance for RAW files (shadows recover)
    max_clip_hi: float = 0.4  # max fraction of the subject blown out
    reject_closed_eyes: bool = True
    closed_eyes_min_faces: int = 1  # reject if at least this many judged faces have closed eyes...
    closed_eyes_max_frac: float = 0.34  # ...and they are more than this fraction of the judged faces
    burst_keep: int = 1  # keep the best N frames of each burst / near-duplicate group
    burst_margin: float = 0.0  # also keep burst frames within this quality of the best (0 = only best N)
    burst_similarity: float = 0.82  # 0..1; higher = only very similar frames count as duplicates
    burst_gap_s: float = 4.0  # frames further apart than this are never duplicates
    keep_at_least_one: bool = True  # a burst whose frames are all only slightly soft (or blinking) keeps its best one

    @property
    def max_blur_px(self) -> float:
        """Blur radius at MEASURE_SIDE allowed in the focus region: strictness 0 -> 1.9 px, 0.5 -> 1.23 px, 1 -> 0.8 px."""
        return float(1.9 * (0.8 / 1.9) ** float(np.clip(self.focus_strictness, 0, 1)))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def schema(cls) -> list[dict]:
        """UI description of every setting: key, label, type ('float'|'int'|'bool'), min, max, step, default, help."""
        d = cls()
        return [{**row, "default": getattr(d, row["key"])} for row in _SCHEMA]

    @classmethod
    def from_json(cls, d: dict) -> "CullSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


_SCHEMA = [
    {"key": "focus_strictness", "label": "Focus strictness", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
     "help": "How sharp the eyes / subject must be. 0 keeps slightly soft frames, 1 only tack-sharp ones."},
    {"key": "max_motion", "label": "Motion blur limit", "type": "float", "min": 0.05, "max": 0.6, "step": 0.01,
     "help": "Reject frames whose blur is this directional (camera shake, moving subject). Lower = stricter."},
    {"key": "exposure_tolerance_ev", "label": "Exposure tolerance (EV)", "type": "float", "min": 0.5, "max": 4.0, "step": 0.1,
     "help": "How many stops the subject may be too dark or too bright before the frame is rejected."},
    {"key": "raw_latitude_ev", "label": "Extra RAW shadow latitude (EV)", "type": "float", "min": 0.0, "max": 3.0, "step": 0.1,
     "help": "RAW files can be pulled up further, so allow this much more underexposure for them."},
    {"key": "max_clip_hi", "label": "Blown highlights limit", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01,
     "help": "Largest fraction of the subject that may be pure white."},
    {"key": "reject_closed_eyes", "label": "Reject closed eyes", "type": "bool", "min": None, "max": None, "step": None,
     "help": "Reject frames where someone is blinking."},
    {"key": "closed_eyes_min_faces", "label": "Closed eyes: min faces", "type": "int", "min": 1, "max": 10, "step": 1,
     "help": "Reject only when at least this many faces have their eyes closed."},
    {"key": "closed_eyes_max_frac", "label": "Closed eyes: max share", "type": "float", "min": 0.0, "max": 1.0, "step": 0.05,
     "help": "...and they are more than this share of the faces whose eyes can be judged."},
    {"key": "burst_keep", "label": "Keep per burst", "type": "int", "min": 1, "max": 10, "step": 1,
     "help": "How many of the best frames to keep from each burst / group of near-duplicates."},
    {"key": "burst_margin", "label": "Burst quality margin", "type": "float", "min": 0.0, "max": 0.5, "step": 0.01,
     "help": "Also keep burst frames whose quality is within this of the best one (0 = only the best N)."},
    {"key": "burst_similarity", "label": "Near-duplicate similarity", "type": "float", "min": 0.5, "max": 0.99, "step": 0.01,
     "help": "How alike two frames must look to count as the same moment. Higher = fewer frames grouped."},
    {"key": "burst_gap_s", "label": "Burst time gap (s)", "type": "float", "min": 0.5, "max": 30.0, "step": 0.5,
     "help": "Frames taken further apart than this are never treated as duplicates."},
    {"key": "keep_at_least_one", "label": "Keep best of a soft burst", "type": "bool", "min": None, "max": None, "step": None,
     "help": "If every frame of a burst is only slightly soft or blinking, keep its best one anyway."},
]


def _t(captured: str | None) -> float | None:
    try:
        return datetime.fromisoformat(captured).timestamp() if captured else None
    except ValueError:
        return None


def group_bursts(scores: list[FrameScore], settings: CullSettings) -> None:
    """Assign ``burst_id`` in place: consecutive (capture order) frames that are close in time and look alike."""
    gid = -1
    for i, s in enumerate(scores):
        match = None
        if s.descriptor:
            d, t1 = np.asarray(s.descriptor), _t(s.captured)
            # look back over the last few frames, so one failed frame (dark, shaken) does not split a burst
            for prev in reversed(scores[max(0, i - 8):i]):
                t0 = _t(prev.captured)
                if t0 is not None and t1 is not None and abs(t1 - t0) > settings.burst_gap_s:
                    break
                if prev.descriptor and similarity(np.asarray(prev.descriptor), d) >= settings.burst_similarity:
                    match = prev
                    break
        if match is None:
            gid += 1
        s.burst_id = match.burst_id if match is not None else gid
    sizes: dict[int, int] = {}
    for s in scores:
        sizes[s.burst_id] = sizes.get(s.burst_id, 0) + 1
    for s in scores:
        s.burst_size = sizes[s.burst_id]


def _ramp(x: float, good: float, bad: float) -> float:
    """1 at ``good``, 0 at ``bad``, smooth in between (a step when they coincide)."""
    if abs(bad - good) < 1e-9:
        return 1.0 if (x <= good if bad >= good else x >= good) else 0.0
    t = float(np.clip((x - good) / (bad - good), 0, 1))
    return 1 - t * t * (3 - 2 * t)


def quality(s: FrameScore, settings: CullSettings) -> tuple[float, dict]:
    lim = settings.max_blur_px
    focus = _ramp(s.focus_blur_px, 0.55 * lim, 1.5 * lim) if s.focus_blur_px is not None else 0.6
    motion = _ramp(s.motion, 0.5 * settings.max_motion, 1.6 * settings.max_motion)
    under = settings.exposure_tolerance_ev + (settings.raw_latitude_ev if s.raw else 0.0)
    ev = s.key_ev
    expo = _ramp(-ev, 0.35 * under, 1.6 * under) if ev < 0 else _ramp(ev, 0.35 * settings.exposure_tolerance_ev, 1.6 * settings.exposure_tolerance_ev)
    expo *= _ramp(s.subject_clip_hi, 0.3 * settings.max_clip_hi, 2.0 * settings.max_clip_hi)
    judged = s.eyes_open + s.eyes_closed
    eyes = 1.0 - 0.6 * (s.eyes_closed / judged) if judged else 1.0
    q = focus**1.0 * motion**0.8 * expo**0.7 * eyes
    return round(float(q), 4), {"focus": round(focus, 3), "stability": round(motion, 3), "exposure": round(expo, 3), "eyes": round(eyes, 3)}


@dataclass
class Decision:
    name: str
    path: str
    keep: bool  # final verdict, overrides applied
    reasons: list[str]
    quality: float  # 0..1
    burst_id: int
    burst_rank: int  # 1 = best frame of its burst
    burst_size: int
    flags: list[str] = field(default_factory=list)  # machine-readable: soft, motion, under, over, clipped, eyes_closed, duplicate
    override: bool | None = None  # manual keep (True) / reject (False); None = automatic
    auto_keep: bool | None = None  # what the culler decided before the override
    focus: float = 0.0
    exposure: float = 0.0
    stability: float = 0.0
    eyes: float = 1.0

    def to_json(self) -> dict:
        return asdict(self)


def _fmt_blur(s: FrameScore) -> str:
    where = {"eyes": "eyes", "faces": "faces", "subject": "subject", "frame": "frame"}[s.focus_region]
    return f"{where} blur {s.focus_blur_px:.1f}px"


def _norm_overrides(overrides) -> dict[str, bool]:
    out = {}
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        out[k] = v if isinstance(v, bool) else str(v).lower() in ("keep", "true", "1", "yes")
    return out


def decide(scores: list[FrameScore], settings: CullSettings | None = None, overrides: dict[str, bool] | None = None) -> list[Decision]:
    """Keep/reject every frame with reasons. Absolute checks first (focus, motion, exposure,
    eyes), then within each burst only the best ``burst_keep`` passing frames survive.

    ``overrides`` maps a file name (or stem) to True (keep) / False (reject); they win over
    the automatic verdict, which stays available as ``Decision.auto_keep``."""
    settings = settings or CullSettings()
    scores = sorted((replace(s) for s in scores), key=lambda s: (s.captured or "", s.name))  # inputs stay untouched
    group_bursts(scores, settings)
    lim = settings.max_blur_px
    rows = []
    for s in scores:
        s.quality, s.sub_scores = quality(s, settings)
        flags, why = [], []
        if s.focus_blur_px is not None and s.focus_blur_px > lim:
            flags.append("soft")
            why.append(f"out of focus ({_fmt_blur(s)} > {lim:.1f}px)")
        if s.motion > settings.max_motion:
            flags.append("motion")
            why.append(f"motion blur (directional {s.motion:.2f} > {settings.max_motion:.2f})")
        under = settings.exposure_tolerance_ev + (settings.raw_latitude_ev if s.raw else 0.0)
        if s.key_ev < -under:
            flags.append("under")
            why.append(f"underexposed by {-s.key_ev:.1f} EV")
        elif s.key_ev > settings.exposure_tolerance_ev:
            flags.append("over")
            why.append(f"overexposed by {s.key_ev:.1f} EV")
        if s.subject_clip_hi > settings.max_clip_hi:
            flags.append("clipped")
            why.append(f"{100 * s.subject_clip_hi:.0f}% of the subject blown out")
        judged = s.eyes_open + s.eyes_closed
        if (settings.reject_closed_eyes and judged and s.eyes_closed >= settings.closed_eyes_min_faces
                and s.eyes_closed / judged > settings.closed_eyes_max_frac):
            flags.append("eyes_closed")
            why.append(f"eyes closed ({s.eyes_closed} of {judged} faces)")
        rows.append((s, flags, why))
    by_burst: dict[int, list] = {}
    for r in rows:
        by_burst.setdefault(r[0].burst_id, []).append(r)
    decisions: dict[str, Decision] = {}
    for gid, members in by_burst.items():
        ranked = sorted(members, key=lambda r: -r[0].quality)
        best_q = ranked[0][0].quality
        kept = 0
        for rank, (s, flags, why) in enumerate(ranked, 1):
            ok = not flags
            reasons = list(why)
            if ok and kept >= settings.burst_keep and not (settings.burst_margin > 0 and s.quality >= best_q - settings.burst_margin):
                ok = False
                flags = flags + ["duplicate"]
                reasons.append(f"near-duplicate: #{rank} of {len(members)} in burst (keeping best {settings.burst_keep})")
            if ok:
                kept += 1
                desc = [_fmt_blur(s) if s.focus_blur_px is not None else "focus n/a"]
                if len(members) > 1:
                    desc.append(f"best of {len(members)}" if rank == 1 else f"#{rank} of {len(members)} in burst")
                reasons = ["keep: " + ", ".join(desc)]
            sub = s.sub_scores
            decisions[s.name] = Decision(s.name, s.path, ok, reasons, s.quality, gid, rank, len(members), flags,
                                         focus=sub["focus"], exposure=sub["exposure"], stability=sub["stability"], eyes=sub["eyes"])
        if kept == 0 and settings.keep_at_least_one:
            s, flags, _ = ranked[0]
            marginal = s.focus_blur_px is None or s.focus_blur_px <= 1.35 * lim
            if flags and set(flags) <= {"soft", "eyes_closed"} and marginal:
                d = decisions[s.name]
                d.keep = True
                d.reasons = [f"keep (best available of {len(members)}, but {'; '.join(d.reasons)})"]
    ov = _norm_overrides(overrides)
    for d in decisions.values():
        d.auto_keep = d.keep
        o = ov.get(d.name, ov.get(Path(d.name).stem))
        if o is not None:
            d.override = o
            if o != d.keep:
                d.reasons = [f"manual {'keep' if o else 'reject'} (auto: {'; '.join(d.reasons)})"]
            d.keep = o
    return [decisions[s.name] for s in scores]


def summarize(decisions: list[Decision]) -> dict:
    flags: dict[str, int] = {}
    for d in decisions:
        if not d.keep:
            for f in d.flags or ["other"]:
                flags[f] = flags.get(f, 0) + 1
    return {"frames": len(decisions), "keep": sum(d.keep for d in decisions), "reject": sum(not d.keep for d in decisions),
            "bursts": len({d.burst_id for d in decisions}), "overridden": sum(d.override is not None for d in decisions),
            "reject_reasons": flags}


def _same_file(a: Path, b: Path) -> bool:
    sa, sb = a.stat(), b.stat()
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def build_album(decisions: list[Decision], out_dir: str | Path, overrides: dict[str, bool | str] | None = None,
                mode: str = "link") -> list[Path]:
    """Create the base album folder from the keepers.

    ``overrides`` maps a file name (or stem) to ``True``/``"keep"`` or ``False``/``"reject"``.
    ``mode="link"`` makes symlinks (no extra disk; falls back to copying where links are not
    allowed), ``"copy"`` copies with metadata. Files this function placed earlier but that are
    no longer kept are removed; anything else in ``out_dir`` is left alone.
    """
    if mode not in ("link", "copy"):
        raise ValueError("mode must be 'link' or 'copy'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm = _norm_overrides(overrides)
    manifest_path = out_dir / _MANIFEST
    try:
        previous = set(json.loads(manifest_path.read_text()).get("files", []))
    except (OSError, ValueError):
        previous = set()
    placed = []
    for d in decisions:
        keep = norm.get(d.name, norm.get(Path(d.name).stem, d.keep))
        if not keep:
            continue
        src, dst = Path(d.path), out_dir / d.name
        if dst.is_symlink() or dst.exists():
            ours = d.name in previous
            linked = dst.is_symlink() and Path(os.readlink(dst)) == src.resolve()
            if linked and mode == "link":
                placed.append(dst)
                continue
            if mode == "copy" and ours and not dst.is_symlink() and _same_file(dst, src):
                placed.append(dst)
                continue
            if not (ours or linked):
                raise FileExistsError(f"{dst} exists and was not created by morph cull")
            dst.unlink()  # switch link <-> copy, or the source changed
        if mode == "link":
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        placed.append(dst)
    names = {p.name for p in placed}
    for old in previous - names:
        p = out_dir / old
        if p.is_symlink() or p.exists():
            p.unlink()
    manifest_path.write_text(json.dumps({"files": sorted(names)}, indent=1))
    return placed


def write_report(path: str | Path, scores: list[FrameScore], decisions: list[Decision], settings: CullSettings) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by = {s.name: s for s in scores}
    report = {"version": SCORE_VERSION, "settings": settings.to_json(), "summary": summarize(decisions),
              "frames": [{**d.to_json(), "score": by[d.name].to_json()} for d in decisions]}
    path.write_text(json.dumps(report, indent=1))
    return path


def cull(dump_dir: str | Path, out_dir: str | Path | None, settings: CullSettings | None = None, cache=None, progress=None,
         people: bool = True, mode: str = "link", overrides=None, report: str | Path | None = None):
    """Score, decide, and (unless ``out_dir`` is None) build the album and write ``cull.json``."""
    settings = settings or CullSettings()
    scores = score_folder(dump_dir, cache, progress, people=people)
    decisions = decide(scores, settings, overrides)
    if out_dir is not None:
        build_album(decisions, out_dir, mode=mode)
    rep = report or (Path(out_dir) / REPORT_NAME if out_dir is not None else Path(cache or Path(dump_dir) / ".morph-cull") / REPORT_NAME)
    write_report(rep, scores, decisions, settings)
    return scores, decisions, Path(rep)
