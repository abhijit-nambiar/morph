"""Deterministic edit recipes: Lightroom-style global sliders plus per-region local adjustments.

All values are normalised so that typical edits fall inside [-1, 1] (exposure is in stops).
A recipe only describes per-pixel tone/colour changes and detail/noise filtering; it can never
move, warp, crop or synthesise image content.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from .segment import REGIONS

HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
GRADE_ZONES = ("shadows", "midtones", "highlights")

TONE_KEYS = ("exposure", "contrast", "highlights", "shadows", "whites", "blacks")
COLOR_KEYS = ("temperature", "tint", "saturation", "vibrance", "hue")
DETAIL_KEYS = ("dehaze", "clarity", "texture", "noise_reduction")

LOCAL_KEYS = TONE_KEYS + ("temperature", "tint", "saturation", "hue") + DETAIL_KEYS
GLOBAL_KEYS = (
    TONE_KEYS
    + COLOR_KEYS
    + DETAIL_KEYS
    + ("curve_shadows", "curve_darks", "curve_lights", "curve_highlights", "sharpen", "grain", "vignette")
    + tuple(f"hsl_{b}_{c}" for b in HSL_BANDS for c in ("hue", "sat", "lum"))
    + tuple(f"grade_{z}_{c}" for z in GRADE_ZONES for c in ("x", "y"))
)

# (min, max) bounds used when fitting; values outside are clamped.
BOUNDS: dict[str, tuple[float, float]] = {k: (-1.0, 1.0) for k in GLOBAL_KEYS}
BOUNDS.update(exposure=(-3.0, 3.0), noise_reduction=(0.0, 1.0), sharpen=(0.0, 1.0), grain=(0.0, 1.0))
LOCAL_BOUNDS: dict[str, tuple[float, float]] = {k: (-1.0, 1.0) for k in LOCAL_KEYS}
LOCAL_BOUNDS.update(exposure=(-2.0, 2.0), noise_reduction=(0.0, 1.0))

LOCAL_REGIONS = tuple(r for r in REGIONS)


def _clamp(v: float, lo: float, hi: float) -> float:
    if v != v:  # NaN -> neutral
        v = 0.0
    return float(min(hi, max(lo, v)))


@dataclass
class Recipe:
    global_: dict[str, float] = field(default_factory=dict)
    local: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def g(self, key: str) -> float:
        return float(self.global_.get(key, 0.0))

    def l(self, region: str, key: str) -> float:
        return float(self.local.get(region, {}).get(key, 0.0))

    def copy(self) -> "Recipe":
        return copy.deepcopy(self)

    def clamped(self) -> "Recipe":
        out = Recipe(meta=dict(self.meta))
        for k, v in self.global_.items():
            if k in BOUNDS:
                out.global_[k] = _clamp(v, *BOUNDS[k])
        for r, d in self.local.items():
            if r in LOCAL_REGIONS:
                out.local[r] = {k: _clamp(v, *LOCAL_BOUNDS[k]) for k, v in d.items() if k in LOCAL_BOUNDS}
        return out

    def pruned(self, eps: float = 2e-3) -> "Recipe":
        out = Recipe({k: v for k, v in self.global_.items() if abs(v) >= eps}, meta=dict(self.meta))
        for r, d in self.local.items():
            kept = {k: v for k, v in d.items() if abs(v) >= eps}
            if kept:
                out.local[r] = kept
        return out

    def plus(self, other: "Recipe", weight: float = 1.0) -> "Recipe":
        out = self.copy()
        for k, v in other.global_.items():
            out.global_[k] = out.global_.get(k, 0.0) + weight * v
        for r, d in other.local.items():
            dst = out.local.setdefault(r, {})
            for k, v in d.items():
                dst[k] = dst.get(k, 0.0) + weight * v
        return out.clamped()

    def scaled(self, weight: float) -> "Recipe":
        return Recipe().plus(self, weight)

    def to_json(self) -> dict:
        r = self.pruned()
        return {
            "global": {k: round(v, 4) for k, v in sorted(r.global_.items())},
            "local": {reg: {k: round(v, 4) for k, v in sorted(d.items())} for reg, d in sorted(r.local.items())},
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict, clamp: bool = True) -> "Recipe":
        """``clamp=False`` for adjustment deltas, which may exceed a slider's range (fitted +0.8 to -1 is -1.8)."""
        r = cls(dict(data.get("global", {})), {k: dict(v) for k, v in data.get("local", {}).items()}, dict(data.get("meta", {})))
        return r.clamped() if clamp else r

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        return cls.from_json(json.loads(Path(path).read_text()))

    def describe(self, limit: int = 12) -> list[str]:
        r = self.pruned(0.02)
        items = [(abs(v), f"{k} {v:+.2f}") for k, v in r.global_.items()]
        items += [(abs(v), f"{reg}.{k} {v:+.2f}") for reg, d in r.local.items() for k, v in d.items()]
        return [s for _, s in sorted(items, reverse=True)[:limit]]


def average(recipes: list[Recipe], weights: list[float] | None = None) -> Recipe:
    if not recipes:
        return Recipe()
    weights = weights or [1.0] * len(recipes)
    total = sum(weights)
    out = Recipe()
    for r, w in zip(recipes, weights):
        out = out.plus(r, w / total)
    return out
