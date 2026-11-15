"""A Morph project: an analysed album, its scenes, and the reference edits that define the look.

Layout (``<project>/``)::

    project.json                 album path, cache path, scenes
    references/<stem>.json       reference recipe, provenance, guard report, manual adjustments
    references/<stem>.target.png the target the recipe was fitted to (working resolution)
    candidates/<stem>/NN.png     generated candidates awaiting selection
    recipes/<stem>.json          final per-photo recipes (after `apply`)
"""

from __future__ import annotations

import json
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .album import Album
from .fit import fit_recipe
from .generate import Candidate, get_generator
from .guard import GuardReport, check
from .imageio import list_images, load_rgb, save_png_u8
from .recipe import Recipe
from .render import render_recipe
from .scenes import Scene, cluster, suggest


@dataclass
class ReferenceEdit:
    name: str
    recipe: Recipe  # fitted (or generator-provided) recipe, before manual adjustments
    adjust: Recipe = field(default_factory=Recipe)  # manual per-region tweaks layered on top
    source: str = ""  # "hand-edit:<path>", "generated:<generator>:<label>", "recipe"
    prompt: str = ""
    guard: dict | None = None
    fit: dict | None = None

    @property
    def final(self) -> Recipe:
        r = self.recipe.plus(self.adjust).clamped().pruned()
        r.meta = {"reference": self.name, "source": self.source}
        return r

    def to_json(self) -> dict:
        return {"name": self.name, "recipe": self.recipe.to_json(), "adjust": self.adjust.to_json(), "source": self.source,
                "prompt": self.prompt, "guard": self.guard, "fit": self.fit}

    @classmethod
    def from_json(cls, d: dict) -> "ReferenceEdit":
        return cls(d["name"], Recipe.from_json(d["recipe"]), Recipe.from_json(d.get("adjust", {}), clamp=False), d.get("source", ""),
                   d.get("prompt", ""), d.get("guard"), d.get("fit"))


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta = json.loads((self.root / "project.json").read_text())
        self.meta = meta
        try:
            self.album = Album.load(meta["album"], meta.get("cache"))
            indexed = set(self.album.names) | {name for name, _ in self.album.skipped}
            files = list_images(meta["album"])
            if (indexed != {p.name for p in files}
                    or any(not Path(p.path).exists() for p in self.album.photos)
                    or any(self.album.source_stat(p) != self.album.stat.get(p.name) for p in files)):
                self.album = Album(meta["album"], meta.get("cache")).analyze()
        except FileNotFoundError:
            # Model/decoder upgrades invalidate analysis, not the user's project or references.
            self.album = Album(meta["album"], meta.get("cache")).analyze()
        self.scenes = [Scene(**s) for s in meta.get("scenes", [])]
        self._sync_scenes()

    def _sync_scenes(self) -> None:
        """Reconcile refreshed analysis while retaining existing scene choices and labels."""
        names = set(self.album.names)
        scenes = []
        used = {s.id for s in self.scenes}
        for scene in self.scenes:
            photos = [n for n in scene.photos if n in names]
            if photos:
                representative = scene.representative if scene.representative in photos else photos[0]
                scenes.append(Scene(scene.id, photos, representative, scene.label))
        covered = {n for s in scenes for n in s.photos}
        added = [i for i, p in enumerate(self.album.photos) if p.name not in covered]
        if added:
            # Cluster only newly discovered photos so established scene boundaries stay intact.
            subset = copy(self.album)
            subset.photos = [self.album.photos[i] for i in added]
            subset.emb_global = self.album.emb_global[added]
            subset.emb_background = self.album.emb_background[added]
            number = 1
            for scene in cluster(subset):
                while f"scene-{number:02d}" in used:
                    number += 1
                scene.id = f"scene-{number:02d}"
                used.add(scene.id)
                scenes.append(scene)
        if [s.to_json() for s in scenes] != [s.to_json() for s in self.scenes]:
            self.scenes = scenes
            self.save()

    # -- creation ------------------------------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, album_dir: str | Path, cache: str | Path | None = None, progress=None, threshold: float = 0.4) -> "Project":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        album = Album(album_dir, cache or root / "cache").analyze(progress=progress)
        scenes = cluster(album, threshold)
        meta = {"album": str(Path(album_dir).resolve()), "cache": str(album.cache.resolve()), "scenes": [s.to_json() for s in scenes]}
        (root / "project.json").write_text(json.dumps(meta, indent=1))
        return cls(root)

    def save(self) -> None:
        self.meta["scenes"] = [s.to_json() for s in self.scenes]
        (self.root / "project.json").write_text(json.dumps(self.meta, indent=1))

    def recluster(self, threshold: float) -> list[Scene]:
        self.scenes = cluster(self.album, threshold)
        self.save()
        return self.scenes

    def scene_of(self, name: str) -> str:
        name = self.album.get(name).name
        return next((s.id for s in self.scenes if name in s.photos), "")

    def suggest(self, k: int | None = None):
        return suggest(self.album, self.scenes, k)

    # -- references ----------------------------------------------------------------------------
    @property
    def ref_dir(self) -> Path:
        return self.root / "references"

    def references(self) -> list[ReferenceEdit]:
        if not self.ref_dir.exists():
            return []
        names = {ph.name for ph in self.album.photos}  # a re-created project may have dropped photos (e.g. re-culled)
        refs = [ReferenceEdit.from_json(json.loads(p.read_text())) for p in sorted(self.ref_dir.glob("*.json"))]
        return [r for r in refs if r.name in names]

    def reference(self, name: str) -> ReferenceEdit | None:
        p = self.ref_dir / f"{Path(self.album.get(name).name).stem}.json"
        return ReferenceEdit.from_json(json.loads(p.read_text())) if p.exists() else None

    def save_reference(self, ref: ReferenceEdit, target: np.ndarray | None = None) -> None:
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(ref.name).stem
        (self.ref_dir / f"{stem}.json").write_text(json.dumps(ref.to_json(), indent=1))
        if target is not None:
            save_png_u8(target, self.ref_dir / f"{stem}.target.png")

    def remove_reference(self, name: str) -> None:
        stem = Path(self.album.get(name).name).stem
        for p in self.ref_dir.glob(f"{stem}.*"):
            p.unlink()

    def candidate_semantics(self, cand: np.ndarray) -> tuple[dict, int]:
        """Re-segment a candidate so the guard can compare regions/faces (catches added/removed/moved objects)."""
        from . import album as album_mod

        if getattr(self, "_segmenter", None) is None:
            self._segmenter = album_mod.Segmenter()
        rm = self._segmenter.segment((np.clip(cand, 0, 1) * 255 + 0.5).astype(np.uint8))
        return rm.masks, len(rm.faces)

    def _inputs(self, name: str):
        img = self.album.work_image(name)
        rm = self.album.masks(name)
        return img, rm

    def reference_from_target(self, name: str, target: np.ndarray, source: str, prompt: str = "",
                              candidate_masks: dict | None = None, faces_cand: int | None = None, allow: tuple = ("pass", "repairable"),
                              semantic: bool = True) -> ReferenceEdit:
        """Guard + fit: turn any target image (hand edit, AI output) into a deterministic recipe."""
        img, rm = self._inputs(name)
        if candidate_masks is None and semantic:
            candidate_masks, faces_cand = self.candidate_semantics(target)
        rep, aligned, conf = check(img, target, rm.masks, candidate_masks, (len(rm.faces), faces_cand) if faces_cand is not None else None)
        if rep.verdict not in allow:
            raise ValueError(f"target rejected by distortion guard: {'; '.join(rep.reasons)}")
        res = fit_recipe(img, aligned, rm.masks, conf)
        ref = ReferenceEdit(self.album.get(name).name, res.recipe, Recipe(), source, prompt, rep.to_json(),
                            {"de_before": round(res.de_before, 3), "de_after": round(res.de_after, 3), "region_de": {k: round(v, 3) for k, v in res.region_de.items()}})
        self.save_reference(ref, aligned)
        return ref

    def reference_from_edit(self, name: str, edit_path: str | Path) -> ReferenceEdit:
        img = self.album.work_image(name)
        target = load_rgb(edit_path, max_side=int(max(img.shape[:2]) * 1.25))
        return self.reference_from_target(name, target, f"hand-edit:{edit_path}")

    def generate(self, name: str, prompt: str, generator: str = "local", n: int = 4, semantic: bool | None = None) -> list[tuple[Candidate, GuardReport]]:
        # The local generator only renders slider recipes, so it cannot move pixels; anything else gets re-segmented.
        semantic = generator != "local" if semantic is None else semantic
        img, rm = self._inputs(name)
        gen = get_generator(generator)
        cands = gen.generate(img, prompt, rm.masks, n=n, stem=Path(self.album.get(name).name).stem)
        out = []
        cdir = self.root / "candidates" / Path(self.album.get(name).name).stem
        cdir.mkdir(parents=True, exist_ok=True)
        for p in cdir.glob("*"):
            p.unlink()
        for i, c in enumerate(cands):
            cm, cf = self.candidate_semantics(c.image) if semantic else (None, None)
            rep, aligned, _ = check(img, c.image, rm.masks, cm, (len(rm.faces), cf) if cf is not None else None)
            c.info["guard"] = rep.to_json()
            c.info["generator"] = generator
            save_png_u8(aligned if rep.verdict != "reject" else c.image, cdir / f"{i:02d}.png")
            meta = {"label": c.label, "prompt": prompt, "generator": generator, "guard": rep.to_json(),
                    "recipe": c.recipe.to_json() if c.recipe else None, "info": {k: v for k, v in c.info.items() if k != "guard"}}
            (cdir / f"{i:02d}.json").write_text(json.dumps(meta, indent=1))
            out.append((c, rep))
        return out

    def pick(self, name: str, index: int) -> ReferenceEdit:
        stem = Path(self.album.get(name).name).stem
        cdir = self.root / "candidates" / stem
        meta = json.loads((cdir / f"{index:02d}.json").read_text())
        source = f"generated:{meta['generator']}:{meta['label']}"
        if meta.get("recipe"):  # slider-based generator: the recipe is exact, no fitting needed
            rec = Recipe.from_json(meta["recipe"])
            img, rm = self._inputs(name)
            ref = ReferenceEdit(self.album.get(name).name, rec, Recipe(), source, meta["prompt"], meta["guard"], {"exact": True})
            self.save_reference(ref, render_recipe(img, rm.masks, rec))
            return ref
        target = load_rgb(cdir / f"{index:02d}.png")
        return self.reference_from_target(name, target, source, meta["prompt"])

    def adjust(self, name: str, changes: dict[str, float]) -> ReferenceEdit:
        """Set manual tweaks, e.g. ``{"exposure": 0.2, "face_skin.texture": -0.3, "background.saturation": -0.2}``."""
        ref = self.reference(name)
        if ref is None:
            ref = ReferenceEdit(self.album.get(name).name, Recipe(), Recipe(), "manual")
        for k, v in changes.items():
            if "." in k:
                reg, key = k.split(".", 1)
                ref.adjust.local.setdefault(reg, {})[key] = float(v)
            else:
                ref.adjust.global_[k] = float(v)
        ref.adjust = ref.adjust.pruned(1e-6)
        self.save_reference(ref)
        return ref
