"""Full-resolution rendering, export verification and the HTML review page."""

from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np

from .album import Album
from .filters import upsample_mask
from .guard import edge_f1, structure_ncc
from .imageio import WRITABLE_SUFFIXES, is_raw, load_rgb, save_image
from .recipe import Recipe
from .render import add_grain, render_recipe


def full_res_masks(album: Album, name: str, img: np.ndarray, recipe: Recipe) -> dict[str, np.ndarray] | None:
    regions = [r for r, d in recipe.local.items() if any(abs(v) > 1e-4 for v in d.values())]
    if not regions:
        return None
    rm = album.masks(name)
    guide = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return {r: upsample_mask(rm.masks[r], guide) for r in regions if r in rm.masks}


def render_photo(album: Album, name: str, recipe: Recipe, max_side: int | None = None,
                 with_grain: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rec = album.get(name)
    img = load_rgb(rec.path, max_side=max_side)
    masks = full_res_masks(album, name, img, recipe)
    seed = int(rec.sha256[:8], 16) % (2**31)
    if with_grain:
        return img, render_recipe(img, masks, recipe, seed=seed)
    r = Recipe(dict(recipe.global_), recipe.local, dict(recipe.meta))
    r.global_.pop("grain", None)
    return img, render_recipe(img, masks, r, seed=seed)


def verify(orig: np.ndarray, out: np.ndarray) -> dict:
    """Non-structural check: same geometry, and local detail still lines up (tone-invariant).

    Run on the grain-free render: grain is seeded per-pixel noise and would only blur the signal."""
    s = 768 / max(orig.shape[:2])
    small = lambda a: cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else a
    ok_shape = orig.shape == out.shape
    a, b = (small(orig), small(out)) if ok_shape else (None, None)
    st = structure_ncc(a, b) if ok_shape else 0.0
    ef = edge_f1(a, b) if ok_shape else 0.0
    return {"same_geometry": bool(ok_shape), "structure": round(st, 4), "edge_f1": round(ef, 4), "ok": bool(ok_shape and st > 0.6)}


def output_name(name: str, fmt: str | None = "jpg") -> str:
    """Export file name: ``fmt`` "jpg" (default, for screens) or "tif" (16-bit, for print); "same"/None keeps the
    source's own format where it is writable (RAW and the rest become JPEG)."""
    p = Path(name)
    if fmt and fmt != "same":
        return f"{p.stem}.{fmt.lower().lstrip('.')}"
    if is_raw(p) or p.suffix.lower() not in WRITABLE_SUFFIXES:
        return f"{p.stem}.jpg"
    return p.name


def output_names(names: list[str], fmt: str | None = "jpg") -> dict[str, str]:
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
                 quality: int = 95, progress=None, fmt: str | None = "jpg", max_bytes: int | None = None) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "photos").mkdir(parents=True, exist_ok=True)
    (out_dir / "recipes").mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)
    report = {}
    files = output_names([p.name for p in album.photos], fmt)
    for i, p in enumerate(album.photos):
        r = recipes[p.name]
        img, out = render_photo(album, p.name, r, max_side, with_grain=False)
        v = verify(img, out)
        if not v["ok"]:
            raise RuntimeError(f"{p.name}: export failed the structure check {v}")
        out = add_grain(out, r.g("grain"), int(album.get(p.name).sha256[:8], 16) % (2**31))
        path = save_image(out, out_dir / "photos" / files[p.name], source=p.path, quality=quality, max_bytes=max_bytes)
        r.save(out_dir / "recipes" / f"{Path(p.name).stem}.json")
        for tag, a in (("before", img), ("after", out)):
            s = 640 / max(a.shape[:2])
            th = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / "preview" / f"{Path(p.name).stem}.{tag}.jpg"), cv2.cvtColor((th * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        report[p.name] = {**v, "size": [out.shape[1], out.shape[0]], "file": files[p.name], "bytes": path.stat().st_size,
                          "over_limit": bool(max_bytes and path.suffix.lower() in (".jpg", ".jpeg", ".webp") and path.stat().st_size > max_bytes)}
        if progress:
            progress(i + 1, len(album.photos), p.name)
    (out_dir / "verification.json").write_text(json.dumps(report, indent=1))
    return report


def _friendly(item: str) -> str:
    """'face_skin.texture -0.40' -> 'Face skin texture -0.40'."""
    key, _, val = item.partition(" ")
    words = key.replace(".", " ").replace("_sat", "_saturation").replace("_lum", "_luminance").replace("hsl_", "").replace("grade_", "grading ").replace("curve_", "curve ").replace("_", " ")
    return f"{words[:1].upper()}{words[1:]} {val}".strip()


def review_page(album: Album, scenes, recipes: dict[str, Recipe], refs: list[str], out_dir: str | Path, title: str = "Morph review") -> Path:
    out_dir = Path(out_dir)
    parts = [f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>:root{{--bg:#0e0f11;--panel:#15171a;--line:#262a30;--text:#e6e8eb;--dim:#9aa1ab;--faint:#878f9a;--accent:#e8b04a}}
body{{font:13.5px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);margin:0;padding:24px 28px}}
h1{{font-size:18px;margin:0 0 4px}}p.lead{{color:var(--dim);margin:0 0 8px}}
h2{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin:28px 0 10px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}figure.ref{{border-color:var(--accent)}}
img{{width:100%;display:block}}.ba{{position:relative;display:block;width:100%;padding:0;border:0;background:#000;cursor:pointer}}
.ba img.after{{position:absolute;inset:0}}.ba.orig img.after,.ba:active img.after{{opacity:0}}.ba:focus-visible{{outline:2px solid var(--accent);outline-offset:-2px}}
.lbl{{position:absolute;left:8px;top:8px;font-size:11px;padding:2px 8px;border-radius:20px;background:rgba(10,11,13,.72);color:var(--dim)}}
.ba .lbl::after{{content:"Edited"}}.ba.orig .lbl::after{{content:"Original"}}
figcaption{{padding:9px 11px;font-size:12px}}figcaption b{{font-weight:600}}.tag{{font-size:10.5px;color:var(--accent);margin-left:6px}}
figcaption .d{{color:var(--faint);margin-top:3px}}</style>
<h1>{html.escape(title)}</h1><p class=lead>Click or press a photo to compare with the original. Outlined photos are the references.</p>"""]
    for s in scenes:
        parts.append(f"<h2>{html.escape(s.id)} &middot; {len(s.photos)} photos</h2><div class=g>")
        for n in s.photos:
            stem = Path(n).stem
            r = recipes.get(n)
            desc = ", ".join(_friendly(d) for d in r.describe(6)) if r else ""
            tag = "<span class=tag>reference</span>" if n in refs else ""
            parts.append(f"""<figure class="{'ref' if n in refs else ''}"><button class=ba aria-label="Toggle original for {html.escape(n)}"><img src="preview/{stem}.before.jpg" alt="" loading=lazy>
<img class=after src="preview/{stem}.after.jpg" alt="{html.escape(n)} edited" loading=lazy><span class=lbl></span></button><figcaption><b>{html.escape(n)}</b>{tag}<div class=d>{html.escape(desc)}</div></figcaption></figure>""")
        parts.append("</div>")
    parts.append("<script>document.addEventListener('click',e=>{const b=e.target.closest('.ba');if(b)b.classList.toggle('orig')})</script>")
    p = out_dir / "index.html"
    p.write_text("\n".join(parts))
    return p
