"""Semantic region masks: people parts (hair, facial skin, body skin, eyes, lips, clothes) and scene parts.

Models (downloaded from the Hugging Face hub on first use, then cached):
  * ``mattmdjaga/segformer_b2_clothes`` - human parsing (hair, face, arms/legs, clothing...)
  * ``jonathandinu/face-parsing``       - face parsing on face crops (skin, eyes, lips, brows...)
  * ``nvidia/segformer-b0-finetuned-ade-512-512`` - scene parsing (sky, vegetation)

All masks are soft float maps in [0, 1] at the analysis resolution. They are later upsampled
with an edge-aware guided filter against the full-resolution photo during rendering.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .filters import guided_filter

REGIONS = (
    "subject",
    "background",
    "sky",
    "vegetation",
    "hair",
    "face_skin",
    "body_skin",
    "eyes",
    "lips",
    "clothes",
)

HUMAN_MODEL = "mattmdjaga/segformer_b2_clothes"
FACE_MODEL = "jonathandinu/face-parsing"
SCENE_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGMENTER_VERSION = "seg-v1"

_HUMAN_GROUPS = {
    "background": ["background"],
    "hair": ["hair"],
    "face": ["face"],
    "glasses": ["sunglasses"],
    "body_skin": ["left-leg", "right-leg", "left-arm", "right-arm"],
    "clothes": ["hat", "upper-clothes", "skirt", "pants", "dress", "belt", "left-shoe", "right-shoe", "bag", "scarf"],
}
_FACE_GROUPS = {
    "skin": ["skin", "nose", "l_ear", "r_ear"],
    "eyes": ["l_eye", "r_eye"],
    "lips": ["mouth", "u_lip", "l_lip"],
    "brows": ["l_brow", "r_brow"],
    "hair": ["hair"],
    "neck": ["neck", "neck_l"],
    "glasses": ["eye_g"],
}
_SKY = ["sky"]
_VEGETATION = ["tree", "grass", "plant", "flower", "palm", "field"]


@dataclass
class RegionMasks:
    masks: dict[str, np.ndarray]
    faces: list[dict] = field(default_factory=list)
    people: list[dict] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.masks.values())).shape

    def coverage(self) -> dict[str, float]:
        return {k: float(v.mean()) for k, v in self.masks.items()}

    def stack(self, names=REGIONS) -> np.ndarray:
        return np.stack([self.masks[n] for n in names], 0)

    def resized(self, size: tuple[int, int]) -> "RegionMasks":
        w, h = size
        return RegionMasks(
            {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA if w < v.shape[1] else cv2.INTER_LINEAR) for k, v in self.masks.items()},
            self.faces,
            self.people,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.clip(np.rint(v * 255), 0, 255).astype(np.uint8) for k, v in self.masks.items()}
        meta = json.dumps({"faces": self.faces, "people": self.people, "version": SEGMENTER_VERSION})
        np.savez_compressed(path, __meta__=np.frombuffer(meta.encode(), dtype=np.uint8), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "RegionMasks":
        data = np.load(path)
        meta = json.loads(bytes(data["__meta__"]).decode())
        masks = {k: data[k].astype(np.float32) / 255.0 for k in data.files if k != "__meta__"}
        return cls(masks, meta.get("faces", []), meta.get("people", []))


def _group_probs(probs: np.ndarray, id2label: dict, groups: dict[str, list[str]]) -> dict[str, np.ndarray]:
    lookup = {str(v).lower(): int(k) for k, v in id2label.items()}
    out = {}
    for name, labels in groups.items():
        ids = [lookup[l] for l in labels if l in lookup]
        out[name] = probs[ids].sum(0) if ids else np.zeros(probs.shape[1:], np.float32)
    return out


class Segmenter:
    """Lazy-loading wrapper around the three parsing models."""

    _lock = threading.Lock()

    def __init__(self, device: str = "cpu", analysis_side: int = 1024, threads: int | None = None):
        self.device = device
        self.analysis_side = analysis_side
        self.threads = threads
        self._models: dict = {}

    def _load(self, name: str):
        with self._lock:
            if name not in self._models:
                import torch
                from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

                if self.threads:
                    torch.set_num_threads(self.threads)
                proc = SegformerImageProcessor.from_pretrained(name)
                model = SegformerForSemanticSegmentation.from_pretrained(name).to(self.device).eval()
                self._models[name] = (proc, model)
            return self._models[name]

    def _probs(self, name: str, img_u8: np.ndarray, out_hw: tuple[int, int] | None = None, resize_to: int | None = None):
        import torch

        proc, model = self._load(name)
        h, w = img_u8.shape[:2]
        x = img_u8
        if resize_to:
            s = resize_to / max(h, w)
            x = cv2.resize(img_u8, (max(32, round(w * s)), max(32, round(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        inputs = proc(images=x, return_tensors="pt", do_resize=False)
        with torch.no_grad():
            logits = model(pixel_values=inputs["pixel_values"].to(self.device)).logits
            target = out_hw or (h, w)
            logits = torch.nn.functional.interpolate(logits, size=target, mode="bilinear", align_corners=False)
            probs = logits.softmax(1)[0].cpu().numpy().astype(np.float32)
        return probs, model.config.id2label

    def segment(self, full_u8: np.ndarray) -> RegionMasks:
        H, W = full_u8.shape[:2]
        s = min(1.0, self.analysis_side / max(H, W))
        aw, ah = max(1, round(W * s)), max(1, round(H * s))
        small = cv2.resize(full_u8, (aw, ah), interpolation=cv2.INTER_AREA) if s < 1 else full_u8.copy()

        probs, labels = self._probs(HUMAN_MODEL, small)
        human = _group_probs(probs, labels, _HUMAN_GROUPS)
        human = self._refine_people(full_u8, small, human, s)

        person = np.clip(1.0 - human["background"], 0, 1)
        maps = {
            "hair": human["hair"].copy(),
            "face_skin": human["face"].copy(),
            "body_skin": human["body_skin"].copy(),
            "clothes": human["clothes"] + human["glasses"],
            "eyes": np.zeros_like(person),
            "lips": np.zeros_like(person),
        }
        faces = self._parse_faces(full_u8, small, human, person, maps, s)

        sprobs, slabels = self._probs(SCENE_MODEL, small, resize_to=640)
        scene = _group_probs(sprobs, slabels, {"sky": _SKY, "vegetation": _VEGETATION})

        guide = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        r = max(2, round(max(ah, aw) / 256))

        def snap(m: np.ndarray) -> np.ndarray:
            return np.clip(guided_filter(guide, np.clip(m, 0, 1), r, 1e-3), 0, 1).astype(np.float32)

        person = snap(person)
        bg = np.clip(1 - person, 0, 1)
        masks = {
            "subject": person,
            "background": bg,
            "sky": snap(scene["sky"]) * bg,
            "vegetation": snap(scene["vegetation"]) * bg,
        }
        for k in ("hair", "face_skin", "body_skin", "eyes", "lips", "clothes"):
            masks[k] = snap(maps[k]) * np.clip(person * 1.15, 0, 1)
        people = _components(person, 0.5, min_frac=0.0015, scale=1 / s)
        return RegionMasks({k: masks[k].astype(np.float32) for k in REGIONS}, faces, people)

    def _refine_people(self, full_u8, small, human, s):
        """Re-parse small or crowded people at higher resolution from the full-size photo."""
        ah, aw = small.shape[:2]
        person = 1.0 - human["background"]
        comps = _components(person, 0.5, min_frac=0.001, scale=1.0)
        acc = {k: np.zeros_like(v) for k, v in human.items()}
        wsum = np.zeros_like(person)
        for c in comps:
            x0, y0, x1, y1 = c["box"]
            bh, bw = y1 - y0, x1 - x0
            if bh > 0.8 * ah and bw > 0.6 * aw:
                continue
            pad_y, pad_x = int(0.08 * bh) + 4, int(0.08 * bw) + 4
            y0, y1 = max(0, y0 - pad_y), min(ah, y1 + pad_y)
            x0, x1 = max(0, x0 - pad_x), min(aw, x1 + pad_x)
            bh, bw = y1 - y0, x1 - x0
            win_w = max(int(bh * 0.8), 16)
            starts = [x0] if bw <= win_w * 1.2 else list(np.linspace(x0, x1 - win_w, max(2, int(np.ceil(2 * bw / win_w)) - 1)).astype(int))
            for sx in starts:
                ex = min(aw, sx + (bw if len(starts) == 1 else win_w))
                fy0, fy1 = int(y0 / s), int(y1 / s)
                fx0, fx1 = int(sx / s), int(ex / s)
                crop = full_u8[fy0:fy1, fx0:fx1]
                if crop.size == 0:
                    continue
                probs, labels = self._probs(HUMAN_MODEL, crop, out_hw=(y1 - y0, ex - sx), resize_to=768)
                g = _group_probs(probs, labels, _HUMAN_GROUPS)
                win = _feather((y1 - y0, ex - sx))
                for k in acc:
                    acc[k][y0:y1, sx:ex] += g[k] * win
                wsum[y0:y1, sx:ex] += win
        if wsum.max() == 0:
            return human
        alpha = np.clip(wsum, 0, 1)
        out = {}
        for k, v in human.items():
            refined = np.where(wsum > 1e-6, acc[k] / np.maximum(wsum, 1e-6), v)
            out[k] = alpha * refined + (1 - alpha) * v
        return out

    def _parse_faces(self, full_u8, small, human, person, maps, s):
        ah, aw = small.shape[:2]
        faces = []
        for c in _components(human["face"], 0.5, min_frac=0.00002, scale=1.0):
            x0, y0, x1, y1 = c["box"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = max(x1 - x0, y1 - y0) * 2.0 + 4
            bx0, by0 = int(max(0, cx - side / 2)), int(max(0, cy - side * 0.55))
            bx1, by1 = int(min(aw, cx + side / 2)), int(min(ah, cy + side * 0.45))
            if bx1 - bx0 < 4 or by1 - by0 < 4:
                continue
            crop = full_u8[int(by0 / s) : int(by1 / s), int(bx0 / s) : int(bx1 / s)]
            if min(crop.shape[:2]) < 12:
                continue
            probs, labels = self._probs(FACE_MODEL, crop, out_hw=(by1 - by0, bx1 - bx0), resize_to=512)
            g = _group_probs(probs, labels, _FACE_GROUPS)
            win = _feather((by1 - by0, bx1 - bx0)) * np.clip(person[by0:by1, bx0:bx1] * 1.2, 0, 1)
            sl = (slice(by0, by1), slice(bx0, bx1))
            maps["face_skin"][sl] = (1 - win) * maps["face_skin"][sl] + win * g["skin"]
            maps["eyes"][sl] = np.maximum(maps["eyes"][sl], win * g["eyes"])
            maps["lips"][sl] = np.maximum(maps["lips"][sl], win * g["lips"])
            maps["hair"][sl] = np.maximum(maps["hair"][sl] * (1 - win * g["skin"]), win * (g["hair"] + g["brows"]))
            maps["body_skin"][sl] = np.maximum(maps["body_skin"][sl], win * g["neck"])
            maps["clothes"][sl] = np.maximum(maps["clothes"][sl] * (1 - win * (g["skin"] + g["neck"])), win * g["glasses"])
            faces.append(
                {
                    "box": [round(x0 / s), round(y0 / s), round(x1 / s), round(y1 / s)],
                    "area_frac": float((x1 - x0) * (y1 - y0) / (aw * ah)),
                    "eyes_px": float(g["eyes"].sum() * (1 / s) ** 2),
                }
            )
        return faces


def _feather(hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    wy = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1) / max(1.0, h * 0.12)
    wx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1) / max(1.0, w * 0.12)
    return np.clip(np.minimum.outer(wy, wx), 0, 1).astype(np.float32)


def _components(prob: np.ndarray, thr: float, min_frac: float, scale: float) -> list[dict]:
    binary = (prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total = prob.shape[0] * prob.shape[1]
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_frac * total:
            continue
        out.append({"box": [round(x * scale), round(y * scale), round((x + w) * scale), round((y + h) * scale)], "area_frac": float(area / total)})
    return out


def overlay(img: np.ndarray, masks: RegionMasks) -> np.ndarray:
    """Colour-coded preview of the person-part masks over a dimmed photo."""
    colors = {
        "sky": (0.3, 0.6, 1.0),
        "vegetation": (0.2, 0.8, 0.2),
        "hair": (0.6, 0.2, 0.8),
        "face_skin": (1.0, 0.6, 0.4),
        "body_skin": (1.0, 0.85, 0.3),
        "eyes": (0.0, 1.0, 1.0),
        "lips": (1.0, 0.0, 0.4),
        "clothes": (0.2, 0.3, 0.9),
    }
    m = masks.resized((img.shape[1], img.shape[0]))
    out = img * 0.45
    for k, c in colors.items():
        a = m.masks[k][..., None] * 0.65
        out = out * (1 - a) + a * np.array(c, np.float32)
    return np.clip(out, 0, 1)
