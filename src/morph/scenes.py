"""Scene grouping and reference-candidate suggestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .album import Album


@dataclass
class Scene:
    id: str
    photos: list[str]
    representative: str
    label: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _lab_vec(rec, region: str) -> np.ndarray | None:
    s = rec.stats.get(region)
    return None if s is None else np.array(s["mean"] + s["std"][:1], np.float32)


def distance_matrix(album: Album, w_bg: float = 0.55, w_glob: float = 0.2, w_color: float = 0.15, w_time: float = 0.1) -> np.ndarray:
    eb, eg = album.emb_background, album.emb_global
    d_bg = np.clip(1 - eb @ eb.T, 0, 2)
    d_gl = np.clip(1 - eg @ eg.T, 0, 2)
    cols = np.stack([(_lab_vec(p, "background") if _lab_vec(p, "background") is not None else _lab_vec(p, "global")) for p in album.photos])
    cols = cols / np.array([25.0, 12.0, 12.0, 15.0], np.float32)
    d_col = np.sqrt(((cols[:, None] - cols[None]) ** 2).sum(-1)) / 4
    times = [datetime.fromisoformat(p.captured).timestamp() if p.captured else None for p in album.photos]
    n = len(times)
    d_t = np.zeros((n, n), np.float32)
    if all(t is not None for t in times):
        t = np.array(times)
        d_t = np.clip(np.abs(t[:, None] - t[None]) / 1800.0, 0, 1)  # 30 min saturates
    for name, D in (("bg", d_bg), ("gl", d_gl), ("col", d_col)):
        med = np.median(D[np.triu_indices(n, 1)]) if n > 1 else 1
        D /= max(med, 1e-6)
    return (w_bg * d_bg + w_glob * d_gl + w_color * d_col + w_time * d_t * 2).astype(np.float64)


def cluster(album: Album, threshold: float = 0.55, min_size: int = 1) -> list[Scene]:
    n = len(album.photos)
    if n == 1:
        return [Scene("scene-01", album.names, album.names[0])]
    D = distance_matrix(album)
    np.fill_diagonal(D, 0)
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold).fit_predict(D)
    groups: dict[int, list[int]] = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)
    ordered = sorted(groups.values(), key=lambda idx: min(idx))
    scenes = []
    q = quality_scores(album)
    for k, idx in enumerate(ordered):
        sub = D[np.ix_(idx, idx)]
        central = -sub.mean(1)
        score = central / (central.std() + 1e-6) * 0.6 + np.array([q[i] for i in idx]) * 0.4 if len(idx) > 2 else np.array([q[i] for i in idx])
        rep = idx[int(np.argmax(score))]
        scenes.append(Scene(f"scene-{k + 1:02d}", [album.photos[i].name for i in idx], album.photos[rep].name))
    return scenes


def quality_scores(album: Album) -> np.ndarray:
    """Z-scored suitability as a reference: sharp subject, faces visible, not clipped, typical exposure."""
    P = album.photos
    sharp = np.log1p([p.subject_sharpness for p in P])
    faces = np.array([min(len(p.faces), 4) for p in P], np.float32)
    face_area = np.array([sum(f.get("area_frac", 0) for f in p.faces) if p.faces else 0 for p in P], np.float32)
    clip = np.array([p.clip_hi * 3 + p.clip_lo for p in P])
    skin = np.array([p.coverage.get("face_skin", 0) + p.coverage.get("body_skin", 0) * 0.5 for p in P])
    z = lambda v: (v - np.mean(v)) / (np.std(v) + 1e-6)
    return 0.35 * z(sharp) + 0.2 * z(faces) + 0.15 * z(np.log1p(face_area + skin * 1e4)) - 0.2 * z(clip) + 0.1 * z(skin)


@dataclass
class Suggestion:
    name: str
    scene: str
    score: float
    coverage_gain: float
    reasons: list[str] = field(default_factory=list)


def suggest(album: Album, scenes: list[Scene], k: int | None = None) -> list[Suggestion]:
    """Greedy facility-location: pick photos that best 'cover' the album (lighting, setting, subjects),
    weighted by how good each is as a reference.  The first picks cover the largest scenes."""
    n = len(album.photos)
    k = k or min(n, max(1, min(len(scenes), 6)))
    D = distance_matrix(album)
    sim = np.exp(-(D**2) / (2 * np.median(D) ** 2 + 1e-9))
    q = quality_scores(album)
    qn = 1 / (1 + np.exp(-q))  # (0, 1)
    scene_of = {nm: s.id for s in scenes for nm in s.photos}
    best = np.zeros(n)
    chosen, out = [], []
    for _ in range(min(k, n)):
        gains = np.array([np.maximum(best, sim[j]).sum() - best.sum() if j not in chosen else -np.inf for j in range(n)])
        score = gains / max(gains[np.isfinite(gains)].max(), 1e-9) * 0.7 + qn * 0.3
        score[chosen] = -np.inf
        j = int(np.argmax(score))
        chosen.append(j)
        p = album.photos[j]
        reasons = [f"covers {gains[j]:.1f} photos' worth of setting/lighting"]
        if p.faces:
            reasons.append(f"{len(p.faces)} face(s) visible")
        if q[j] > 0.5:
            reasons.append("sharp, well exposed subject")
        if p.clip_hi > 0.02:
            reasons.append("some clipped highlights")
        reasons.append(f"scene {scene_of.get(p.name, '?')}")
        out.append(Suggestion(p.name, scene_of.get(p.name, "?"), round(float(score[j]), 4), round(float(gains[j]), 3), reasons))
        best = np.maximum(best, sim[j])
    return out


def adjusted_rand(a: dict[str, str], b: dict[str, str]) -> float:
    from sklearn.metrics import adjusted_rand_score

    keys = sorted(set(a) & set(b))
    return float(adjusted_rand_score([a[k] for k in keys], [b[k] for k in keys]))
