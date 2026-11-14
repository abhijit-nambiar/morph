"""Pluggable target generators: (original image, prompt, masks) -> several candidate edits.

* ``local``   - Morph's built-in stand-in: parses the look into sliders, adapts them to the photo
                (auto tone toward a professional baseline) and renders N variations.
* ``openai``  - OpenAI image-edit API (``OPENAI_API_KEY``); sends the original + prompt.
* ``http:URL``- any service accepting ``{"image": <b64 png>, "prompt": str, "n": int}`` and
                returning ``{"images": [<b64 png>, ...]}``.
* ``files:DIR``- pre-made candidates on disk named ``<stem>*.{jpg,png}`` (e.g. exported from any editor).

Every candidate goes through :mod:`morph.guard` before it can be used as a reference.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .look import parse
from .recipe import Recipe
from .render import render_recipe

STRUCTURE_INSTRUCTION = (
    "Edit only colour, white balance, exposure, contrast, tone, saturation and noise. Do not move, add, "
    "remove, reshape, crop or retouch anything; keep every edge and object exactly where it is."
)


@dataclass
class Candidate:
    image: np.ndarray
    label: str
    recipe: Recipe | None = None  # known when the generator is slider-based
    info: dict = field(default_factory=dict)


class Generator:
    name = "base"

    def generate(self, image: np.ndarray, prompt: str, masks: dict[str, np.ndarray] | None, n: int = 4, stem: str = "") -> list[Candidate]:
        raise NotImplementedError


def auto_tone(image: np.ndarray, masks: dict[str, np.ndarray] | None) -> Recipe:
    """A neutral professional baseline: expose the subject properly, protect highlights, open shadows."""
    from .colorspace import srgb_to_lab

    lab = srgb_to_lab(image)
    L = lab[..., 0]
    w = None
    if masks is not None:
        face = masks.get("face_skin")
        if face is not None and face.sum() > 0.002 * face.size:
            w = face
        elif masks.get("subject") is not None and masks["subject"].sum() > 0.02 * L.size:
            w = masks["subject"]
    key = float((L * w).sum() / w.sum()) if w is not None else float(np.median(L))
    goal = 66.0 if w is not None and masks.get("face_skin") is not None and w is masks.get("face_skin") else 52.0
    ev = float(np.clip((goal - key) / 22.0, -1.0, 1.0))
    hi, lo = float((L > 97).mean()), float((L < 4).mean())
    a, b = float(np.median(lab[..., 1])), float(np.median(lab[..., 2]))
    r = Recipe({
        "exposure": ev * 0.8,
        "highlights": -min(0.6, hi * 8 + 0.1),
        "shadows": min(0.5, lo * 6 + 0.1),
        "temperature": float(np.clip(-b / 60.0, -0.25, 0.25)) * 0.5,
        "tint": float(np.clip(-a / 60.0, -0.2, 0.2)) * 0.5,
        "vibrance": 0.1,
    }, meta={"auto_tone": {"key": key, "goal": goal}})
    return r.pruned()


class LocalGenerator(Generator):
    """Devin's stand-in for an AI editor: interpret the words, adapt to the photo, offer variations."""

    name = "local"
    VARIANTS = (("as described", 1.0, 0.0), ("stronger", 1.35, 0.0), ("subtler", 0.7, 0.0), ("warmer", 1.0, 0.12), ("cooler", 1.0, -0.12))

    def generate(self, image, prompt, masks, n=4, stem=""):
        look = parse(prompt)
        base = auto_tone(image, masks)
        out = []
        for label, strength, dtemp in self.VARIANTS[: max(1, n)]:
            r = base.plus(look.recipe, strength)
            if dtemp:
                r.global_["temperature"] = r.g("temperature") + dtemp
            r = r.clamped().pruned()
            r.meta = {"look": prompt, "variant": label, "matched": look.matched, "unknown": look.unknown}
            out.append(Candidate(render_recipe(image, masks, r), label, r, {"matched": look.matched, "unknown": look.unknown}))
        return out


def _b64png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
    return im  # geometry differences are resolved by guard.align


class HTTPGenerator(Generator):
    name = "http"

    def __init__(self, url: str, headers: dict | None = None):
        self.url, self.headers = url, headers or {}

    def generate(self, image, prompt, masks, n=4, stem=""):
        import requests

        payload = {"image": _b64png(image), "prompt": f"{prompt}\n\n{STRUCTURE_INSTRUCTION}", "n": n}
        r = requests.post(self.url, json=payload, headers=self.headers, timeout=300)
        r.raise_for_status()
        return [Candidate(_decode(base64.b64decode(b), image.shape[:2]), f"api #{i + 1}") for i, b in enumerate(r.json()["images"])]


class OpenAIGenerator(Generator):
    """OpenAI ``images/edits`` (gpt-image-1).  Requires ``OPENAI_API_KEY``."""

    name = "openai"

    def __init__(self, model: str = "gpt-image-1", api_key: str | None = None):
        self.model = model
        self.key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def generate(self, image, prompt, masks, n=4, stem=""):
        import requests

        h, w = image.shape[:2]
        size = "1024x1536" if h > w * 1.2 else "1536x1024" if w > h * 1.2 else "1024x1024"
        files = {"image": ("image.png", base64.b64decode(_b64png(image)), "image/png")}
        data = {"model": self.model, "prompt": f"{prompt}\n\n{STRUCTURE_INSTRUCTION}", "n": str(n), "size": size, "input_fidelity": "high"}
        r = requests.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {self.key}"}, files=files, data=data, timeout=600)
        r.raise_for_status()
        return [Candidate(_decode(base64.b64decode(d["b64_json"]), (h, w)), f"openai #{i + 1}") for i, d in enumerate(r.json()["data"])]


class FileGenerator(Generator):
    name = "files"

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def generate(self, image, prompt, masks, n=4, stem=""):
        from .imageio import load_rgb

        files = sorted(p for p in self.folder.glob(f"{stem}*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        h, w = image.shape[:2]
        return [Candidate(load_rgb(p, max_side=max(h, w)), p.name, info={"path": str(p)}) for p in files[:n]]


def get_generator(spec: str) -> Generator:
    if spec in ("local", "devin"):
        return LocalGenerator()
    if spec == "openai":
        return OpenAIGenerator()
    if spec.startswith("http:") or spec.startswith("https:"):
        return HTTPGenerator(spec if "//" in spec else spec.split(":", 1)[1])
    if spec.startswith("files:"):
        return FileGenerator(spec.split(":", 1)[1])
    raise ValueError(f"unknown generator {spec!r} (use local, openai, http(s)://..., files:DIR)")
