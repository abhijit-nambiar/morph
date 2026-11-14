"""Full-resolution rendering, export verification and the HTML review page."""

from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np

from .album import Album
from .filters import upsample_mask
from .guard import edge_f1
from .imageio import RASTER_SUFFIXES, is_raw, load_rgb, save_image
from .recipe import Recipe
from .render import render_recipe


def full_res_masks(album: Album, name: str, img: np.ndarray, recipe: Recipe) -> dict[str, np.ndarray] | None:
    regions = [r for r, d in recipe.local.items() if any(abs(v) > 1e-4 for v in d.values())]
    if not regions:
        return None
    rm = album.masks(name)
    guide = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return {r: upsample_mask(rm.masks[r], guide) for r in regions if r in rm.masks}


def render_photo(album: Album, name: str, recipe: Recipe, max_side: int | None = None) -> np.ndarray:
    rec = album.get(name)
    img = load_rgb(rec.path, max_side=max_side)
    masks = full_res_masks(album, name, img, recipe)
    return img, render_recipe(img, masks, recipe, seed=int(rec.sha256[:8], 16) % (2**31))


def verify(orig: np.ndarray, out: np.ndarray) -> dict:
    """Non-structural check: same geometry, edges in the same places."""
    s = 768 / max(orig.shape[:2])
    small = lambda a: cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else a
    ok_shape = orig.shape == out.shape
    ef = edge_f1(small(orig), small(out)) if ok_shape else 0.0
    return {"same_geometry": bool(ok_shape), "edge_f1": round(ef, 4), "ok": bool(ok_shape and ef > 0.8)}


def output_name(name: str, fmt: str | None = None) -> str:
    """Export file name: ``fmt`` ("jpg"/"tif") if given, else the source's own format; RAW sources become JPEG."""
    p = Path(name)
    if fmt:
        return f"{p.stem}.{fmt.lower().lstrip('.')}"
    if is_raw(p) or p.suffix.lower() not in RASTER_SUFFIXES:
        return f"{p.stem}.jpg"
    return p.name


def output_names(names: list[str], fmt: str | None = None) -> dict[str, str]:
    """``output_name`` for a whole album, disambiguating sources that would land on the same file
    (``a.jpg`` + ``a.png`` with ``fmt="jpg"``, or a RAW+JPEG pair): the one already named like its
    output keeps it, the others get their source format in the name (``a-png.jpg``); names that still
    collide case-insensitively are numbered (``a.JPG`` + ``a.jpg`` -> ``a.JPG`` + ``a-2.jpg``)."""
    out = {n: output_name(n, fmt) for n in names}
    groups: dict[str, list[str]] = {}
    for n, o in out.items():
        groups.setdefault(o.lower(), []).append(n)  # case-insensitive: macOS / Windows file systems
    for members in groups.values():
        if len(members) < 2:
            continue
        for n in members:
            if out[n] != n:
                o = Path(out[n])
                out[n] = f"{o.stem}-{Path(n).suffix.lstrip('.').lower()}{o.suffix}"
    taken: set[str] = set()
    wanted = {o.lower() for o in out.values()}
    for n in sorted(names, key=lambda n: out[n] != n):  # still colliding (a.JPG + a.jpg): number the later
        o, k = Path(out[n]), 2
        if out[n].lower() in taken:
            while f"{o.stem}-{k}{o.suffix}".lower() in taken | wanted:
                k += 1
            out[n] = f"{o.stem}-{k}{o.suffix}"
        taken.add(out[n].lower())
    return out


def export_album(album: Album, recipes: dict[str, Recipe], out_dir: str | Path, max_side: int | None = None,
                 quality: int = 95, progress=None, fmt: str | None = None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)
    (out_dir / "recipes").mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)
    report = {}
    files = output_names([p.name for p in album.photos], fmt)
    for i, p in enumerate(album.photos):
        r = recipes[p.name]
        img, out = render_photo(album, p.name, r, max_side)
        v = verify(img, out)
        if not v["ok"]:
            raise RuntimeError(f"{p.name}: export failed the structure check {v}")
        save_image(out, out_dir / "photos" / files[p.name], source=p.path, quality=quality)
        r.save(out_dir / "recipes" / f"{Path(p.name).stem}.json")
        for tag, a in (("before", img), ("after", out)):
            s = 640 / max(a.shape[:2])
            th = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / "preview" / f"{Path(p.name).stem}.{tag}.jpg"), cv2.cvtColor((th * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        report[p.name] = {**v, "size": [out.shape[1], out.shape[0]], "file": files[p.name]}
        if progress:
            progress(i + 1, len(album.photos), p.name)
    (out_dir / "verification.json").write_text(json.dumps(report, indent=1))
    return report


def review_page(album: Album, scenes, recipes: dict[str, Recipe], refs: list[str], out_dir: str | Path, title: str = "Morph review") -> Path:
    out_dir = Path(out_dir)
    parts = [f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<style>body{{font:14px system-ui;background:#111;color:#ddd;margin:24px}}h2{{margin-top:32px}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
figure{{margin:0;background:#1b1b1b;padding:8px;border-radius:8px}}figure.ref{{outline:2px solid #e8b04a}}img{{width:100%;display:block;border-radius:4px}}
figcaption{{font-size:12px;color:#aaa;margin-top:6px}}.ba{{position:relative}}.ba img.after{{position:absolute;inset:0}}.ba:hover img.after{{opacity:0}}
code{{color:#9cf}}</style><h1>{html.escape(title)}</h1><p>Hover a photo to see the original. Gold outline = reference edit.</p>"""]
    for s in scenes:
        parts.append(f"<h2>{html.escape(s.id)} &middot; {len(s.photos)} photos</h2><div class=g>")
        for n in s.photos:
            stem = Path(n).stem
            r = recipes.get(n)
            desc = ", ".join(r.describe(6)) if r else ""
            parts.append(f"""<figure class="{'ref' if n in refs else ''}"><div class=ba><img src="preview/{stem}.before.jpg" loading=lazy>
<img class=after src="preview/{stem}.after.jpg" loading=lazy></div><figcaption><b>{html.escape(n)}</b><br><code>{html.escape(desc)}</code></figcaption></figure>""")
        parts.append("</div>")
    p = out_dir / "index.html"
    p.write_text("\n".join(parts))
    return p
