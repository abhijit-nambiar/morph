"""Album analysis: per-photo metadata, working-resolution copies, region masks, embeddings, stats.

Everything is cached under ``<cache>/`` keyed by the source file's SHA-256 so re-runs are cheap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .colorspace import srgb_to_lab
from .imageio import RAW_RENDER_VERSION, PhotoInfo, list_images, load_rgb, read_info, save_png_u8
from .segment import REGIONS, SEGMENTER_VERSION, RegionMasks, Segmenter

log = logging.getLogger(__name__)

WORK_SIDE = 1024
EMBED_MODEL = "facebook/dinov2-small"
ANALYSIS_VERSION = f"a2-icc-{SEGMENTER_VERSION}-{RAW_RENDER_VERSION}"
STAT_REGIONS = ("subject", "background", "sky", "hair", "face_skin", "body_skin", "clothes", "eyes")


class Embedder:
    """DINOv2 features: a whole-image vector and a background-only (people masked out) vector."""

    def __init__(self, threads: int | None = None):
        import torch

        if threads:
            torch.set_num_threads(threads)
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModel

            self._proc = AutoImageProcessor.from_pretrained(EMBED_MODEL)
            self._model = AutoModel.from_pretrained(EMBED_MODEL).eval()

    def __call__(self, img: np.ndarray, background: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self._load()
        side = 448
        h, w = img.shape[:2]
        s = side / max(h, w)
        nh, nw = max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)
        x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        mean, std = np.array(self._proc.image_mean), np.array(self._proc.image_std)
        t = torch.from_numpy(((x - mean) / std).astype(np.float32)).permute(2, 0, 1)[None]
        with torch.no_grad():
            hs = self._model(pixel_values=t).last_hidden_state[0].numpy()
        cls, patches = hs[0], hs[1:]
        gh, gw = nh // 14, nw // 14
        patches = patches[-gh * gw :]
        if background is None:
            wts = np.ones(gh * gw, np.float32)
        else:
            wts = cv2.resize(background.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA).reshape(-1) ** 2
            if wts.sum() < 1e-3 * len(wts):
                wts = np.ones_like(wts)
        bg = (patches * wts[:, None]).sum(0) / wts.sum()
        glob = np.concatenate([cls, patches.mean(0)])
        n = lambda v: (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
        return n(glob), n(bg)


def region_stats(img: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    """Weighted Lab mean / std and luminance percentiles per region (on a working-res image)."""
    lab = srgb_to_lab(img).reshape(-1, 3)
    out = {}
    for r in ("global",) + STAT_REGIONS:
        w = np.ones(lab.shape[0], np.float32) if r == "global" else masks[r].reshape(-1)
        tot = float(w.sum())
        if tot < 50:
            continue
        mu = (lab * w[:, None]).sum(0) / tot
        sd = np.sqrt(((lab - mu) ** 2 * w[:, None]).sum(0) / tot)
        sel = lab[w > 0.5, 0] if (w > 0.5).sum() > 50 else lab[:, 0]
        p = np.percentile(sel, [5, 50, 95])
        out[r] = {"mean": mu.round(3).tolist(), "std": sd.round(3).tolist(), "pct": p.round(3).tolist(), "coverage": round(tot / len(w), 5)}
    return out


def sharpness(img: np.ndarray, weight: np.ndarray | None = None) -> float:
    g = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    if weight is None or weight.sum() < 100:
        return float(lap.var())
    w = weight / weight.sum()
    mu = (lap * w).sum()
    return float(((lap - mu) ** 2 * w).sum())


@dataclass
class PhotoRecord:
    name: str
    path: str
    width: int
    height: int
    sha256: str
    captured: str | None
    ev: float | None
    work_size: tuple[int, int]
    faces: list = field(default_factory=list)
    people: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    sharpness: float = 0.0
    subject_sharpness: float = 0.0
    clip_hi: float = 0.0
    clip_lo: float = 0.0

    @property
    def stem(self) -> str:
        return Path(self.name).stem


class Album:
    def __init__(self, folder: str | Path, cache: str | Path | None = None):
        self.folder = Path(folder)
        self.cache = Path(cache) if cache else self.folder / ".morph-cache"
        self.photos: list[PhotoRecord] = []
        self.skipped: list[tuple[str, str]] = []
        self.stat: dict[str, list[int]] = {}
        self.emb_global: np.ndarray | None = None
        self.emb_background: np.ndarray | None = None

    # -- cache paths ---------------------------------------------------------------------------
    def work_path(self, rec: PhotoRecord) -> Path:
        return self.cache / "work" / f"{rec.sha256[:16]}.png"

    def mask_path(self, rec: PhotoRecord) -> Path:
        return self.cache / "masks" / f"{rec.sha256[:16]}.npz"

    def work_image(self, name: str) -> np.ndarray:
        rec = self.get(name)
        im = cv2.imread(str(self.work_path(rec)), cv2.IMREAD_COLOR)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def masks(self, name: str) -> RegionMasks:
        return RegionMasks.load(self.mask_path(self.get(name)))

    def get(self, name: str) -> PhotoRecord:
        for p in self.photos:
            if p.name == name or p.stem == name:
                return p
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.photos]

    # -- analysis ------------------------------------------------------------------------------
    @staticmethod
    def source_stat(path: str | Path) -> list[int]:
        stat = Path(path).stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

    def analyze(self, segmenter: Segmenter | None = None, embedder: Embedder | None = None, progress=None) -> "Album":
        self.cache.mkdir(parents=True, exist_ok=True)
        prior_items = self._load_index()
        prior = {t[0].sha256: t for t in prior_items}
        prior_names = {t[0].name: t for t in prior_items}
        prior_stat, prior_skipped = self.stat, self.skipped
        segmenter = segmenter or Segmenter()
        embedder = embedder or Embedder()
        files = list_images(self.folder)
        if not files:
            raise FileNotFoundError(f"no images found in {self.folder}")
        recs, eg, eb = [], [], []
        self.skipped = []
        self.stat = {}
        visited = set()
        complete = False
        try:
            for i, f in enumerate(files):
                try:
                    self.stat[f.name] = self.source_stat(f)
                    old = prior_names.get(f.name) if self.stat[f.name] == prior_stat.get(f.name) else None
                    if old is None or not self.mask_path(old[0]).exists() or not self.work_path(old[0]).exists():
                        info = read_info(f)
                        old = prior.get(info.sha256)
                    if old is not None and self.mask_path(old[0]).exists() and self.work_path(old[0]).exists():
                        rec, g, b = replace(old[0], name=f.name, path=str(f.resolve())), old[1], old[2]
                    else:
                        rec, g, b = self._analyze_one(info, segmenter, embedder)
                except Exception as e:  # noqa: BLE001 - a corrupt card file must not sink the album
                    self.skipped.append((f.name, f"{type(e).__name__}: {e}"))
                    visited.add(f.name)
                    if progress:
                        progress(i + 1, len(files), f.name)
                    continue
                recs.append(rec)
                eg.append(g)
                eb.append(b)
                visited.add(f.name)
                if progress:
                    progress(i + 1, len(files), f.name)
            complete = True
        finally:
            if not complete:
                for rec, g, b in prior_items:
                    if rec.name in visited:
                        continue
                    try:
                        valid = (self.source_stat(self.folder / rec.name) == prior_stat.get(rec.name)
                                 and self.mask_path(rec).exists() and self.work_path(rec).exists())
                    except OSError:
                        valid = False
                    if valid:
                        recs.append(rec)
                        eg.append(g)
                        eb.append(b)
                        self.stat[rec.name] = prior_stat[rec.name]
                for name, reason in prior_skipped:
                    if name not in visited:
                        try:
                            if self.source_stat(self.folder / name) == prior_stat.get(name):
                                self.skipped.append((name, reason))
                                self.stat[name] = prior_stat[name]
                        except OSError:
                            pass
            known = {rec.name for rec in recs} | {name for name, _ in self.skipped}
            self.stat = {name: value for name, value in self.stat.items() if name in known}
            if recs:
                recs_order = sorted(range(len(recs)), key=lambda i: (recs[i].captured or "", recs[i].name))
                self.photos = [recs[i] for i in recs_order]
                self.emb_global = np.stack([eg[i] for i in recs_order])
                self.emb_background = np.stack([eb[i] for i in recs_order])
                self._save_index()
        if not recs:
            raise FileNotFoundError(f"no readable images in {self.folder}: " + "; ".join(f"{n} ({r})" for n, r in self.skipped))
        return self

    def _analyze_one(self, info: PhotoInfo, segmenter: Segmenter, embedder: Embedder):
        img = load_rgb(info.path, max_side=WORK_SIDE)
        u8 = (img * 255 + 0.5).astype(np.uint8)
        rm = segmenter.segment(u8)
        h, w = img.shape[:2]
        if rm.shape != (h, w):
            rm = rm.resized((w, h))
        g, b = embedder(img, rm.masks["background"])
        L = srgb_to_lab(img)[..., 0]
        rec = PhotoRecord(
            name=info.name,
            path=str(Path(info.path).resolve()),
            width=info.width,
            height=info.height,
            sha256=info.sha256,
            captured=info.captured.isoformat() if info.captured else None,
            ev=info.exposure_value,
            work_size=(w, h),
            faces=rm.faces,
            people=rm.people,
            coverage={k: round(v, 5) for k, v in rm.coverage().items()},
            stats=region_stats(img, rm.masks),
            sharpness=sharpness(img),
            subject_sharpness=sharpness(img, rm.masks["face_skin"] + rm.masks["eyes"] * 4 if rm.masks["face_skin"].sum() > 200 else rm.masks["subject"]),
            clip_hi=float((L > 98).mean()),
            clip_lo=float((L < 2).mean()),
        )
        save_png_u8(img, self.work_path(rec))
        rm.save(self.mask_path(rec))
        return rec, g, b

    def _index_path(self) -> Path:
        return self.cache / "analysis.json"

    def _save_index(self) -> None:
        data = {"version": ANALYSIS_VERSION, "folder": str(self.folder), "photos": [asdict(p) for p in self.photos],
                "skipped": self.skipped, "stat": self.stat}
        self._index_path().write_text(json.dumps(data, indent=1))
        np.savez(self.cache / "embeddings.npz", glob=self.emb_global, bg=self.emb_background, sha=np.array([p.sha256 for p in self.photos]))

    def _load_index(self) -> list[tuple[PhotoRecord, np.ndarray, np.ndarray]]:
        try:
            data = json.loads(self._index_path().read_text())
            if data.get("version") != ANALYSIS_VERSION:
                return []
            emb = np.load(self.cache / "embeddings.npz")
            if len(emb["glob"]) != len(data["photos"]):
                return []
            out = []
            for i, p in enumerate(data["photos"]):
                p["work_size"] = tuple(p["work_size"])
                out.append((PhotoRecord(**p), emb["glob"][i], emb["bg"][i]))
            self.skipped = [tuple(item) for item in data.get("skipped", [])]
            self.stat = data.get("stat", {})
            return out
        except Exception:  # noqa: BLE001 - any broken cache just means re-analysis
            return []

    @classmethod
    def load(cls, folder: str | Path, cache: str | Path | None = None) -> "Album":
        a = cls(folder, cache)
        prior = a._load_index()
        if not prior:
            raise FileNotFoundError(f"album not analysed yet: run `morph analyze {folder}`")
        items = sorted(prior, key=lambda t: (t[0].captured or "", t[0].name))
        a.photos = [t[0] for t in items]
        a.emb_global = np.stack([t[1] for t in items])
        a.emb_background = np.stack([t[2] for t in items])
        return a
