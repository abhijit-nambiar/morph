"""Morph Studio: a small local web app for building reference edits and running the album.

Flow: pick a suggested photo -> prompt a generator for candidates (each checked by the distortion
guard) or upload your own edit -> pick one -> fine-tune global / per-region sliders with a live
preview -> repeat for a few photos -> apply to the whole album and open the review page.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from PIL import Image

from .project import Project
from .recipe import GLOBAL_KEYS, LOCAL_KEYS, Recipe
from .render import render_recipe
from .segment import REGIONS, overlay

PREVIEW_SIDE = 900


def _jpeg(img: np.ndarray, side: int | None = PREVIEW_SIDE, q: int = 88) -> bytes:
    if side and max(img.shape[:2]) > side:
        s = side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, "JPEG", quality=q)
    return buf.getvalue()


class State:
    def __init__(self, root: str | Path):
        self.project = Project(root)
        self.lock = threading.Lock()
        self.job = {"state": "idle"}
        self._suggest = None
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def inputs(self, name: str):
        if name not in self._cache:
            img = self.project.album.work_image(name)
            s = PREVIEW_SIDE / max(img.shape[:2])
            rm = self.project.album.masks(name)
            size = (round(img.shape[1] * s), round(img.shape[0] * s))
            self._cache[name] = (cv2.resize(img, size, interpolation=cv2.INTER_AREA), rm.resized(size).masks)
        return self._cache[name]

    def suggestions(self):
        if self._suggest is None:
            self._suggest = self.project.suggest()
        return self._suggest

    def state(self) -> dict:
        p = self.project
        refs = {r.name: r for r in p.references()}
        return {
            "scenes": [s.to_json() for s in p.scenes],
            "photos": {ph.name: {"faces": len(ph.faces), "scene": p.scene_of(ph.name), "size": [ph.width, ph.height]} for ph in p.album.photos},
            "suggestions": [s.__dict__ for s in self.suggestions()],
            "references": {n: {"source": r.source, "prompt": r.prompt, "guard": r.guard, "fit": r.fit, "final": r.final.to_json(), "adjust": r.adjust.to_json()} for n, r in refs.items()},
            "regions": ["global"] + list(REGIONS),
            "global_keys": list(GLOBAL_KEYS),
            "local_keys": list(LOCAL_KEYS),
            "job": self.job,
        }


HTML = (Path(__file__).parent / "studio.html")


class Handler(BaseHTTPRequestHandler):
    S: State = None  # type: ignore

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), code=code)

    def do_GET(self):
        try:
            self._get()
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            self._post(json.loads(self.rfile.read(n) or b"{}"))
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _get(self):
        S, u = self.S, urlparse(self.path)
        parts = [unquote(x) for x in u.path.strip("/").split("/")]
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self._send(HTML.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._json(S.state())
        if parts[0] == "img" and len(parts) >= 3:
            kind, name = parts[1], parts[2]
            if kind == "thumb":
                img, _ = S.inputs(name)
                return self._send(_jpeg(img, 240, 80), "image/jpeg")
            if kind == "orig":
                return self._send(_jpeg(S.inputs(name)[0]), "image/jpeg")
            if kind == "masks":
                img, m = S.inputs(name)
                from .segment import RegionMasks

                return self._send(_jpeg(overlay(img, RegionMasks(m))), "image/jpeg")
            if kind == "ref":
                ref = S.project.reference(name)
                img, m = S.inputs(name)
                return self._send(_jpeg(render_recipe(img, m, ref.final if ref else Recipe())), "image/jpeg")
            if kind == "cand":
                stem = Path(S.project.album.get(name).name).stem
                f = S.project.root / "candidates" / stem / f"{int(parts[3]):02d}.png"
                return self._send(_jpeg(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255), "image/jpeg")
        if parts[0] == "out" and S.job.get("out"):
            f = (Path(S.job["out"]) / "/".join(parts[1:])).resolve()
            if f.is_file() and str(f).startswith(str(Path(S.job["out"]).resolve())):
                ctype = {".html": "text/html", ".jpg": "image/jpeg", ".json": "application/json"}.get(f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
        self._json({"error": "not found"}, 404)

    def _post(self, b):
        S, path = self.S, urlparse(self.path).path
        P = S.project
        if path == "/api/generate":
            with S.lock:
                res = P.generate(b["photo"], b["prompt"], b.get("generator", "local"), int(b.get("n", 4)))
            return self._json({"candidates": [{"label": c.label, "guard": rep.to_json(), "info": {k: v for k, v in c.info.items() if k != "guard"}} for c, rep in res]})
        if path == "/api/pick":
            with S.lock:
                ref = P.pick(b["photo"], int(b["index"]))
            return self._json({"ok": True, "source": ref.source})
        if path == "/api/upload":
            raw = base64.b64decode(b["data"].split(",", 1)[-1])
            img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.float32) / 255
            with S.lock:
                ref = P.reference_from_target(b["photo"], img, f"hand-edit:{b.get('filename', 'upload')}")
            return self._json({"ok": True, "guard": ref.guard, "fit": ref.fit})
        if path == "/api/preview":
            ref = P.reference(b["photo"])
            base = ref.recipe if ref else Recipe()
            adj = Recipe.from_json(b.get("adjust", {}))
            img, m = S.inputs(b["photo"])
            out = render_recipe(img, m, base.plus(adj).clamped())
            return self._send(_jpeg(out), "image/jpeg")
        if path == "/api/adjust":
            with S.lock:
                ref = P.reference(b["photo"])
                if ref is not None:
                    ref.adjust = Recipe()
                    P.save_reference(ref)
                changes = {}
                adj = Recipe.from_json(b.get("adjust", {}))
                changes.update(adj.global_)
                for reg, d in adj.local.items():
                    changes.update({f"{reg}.{k}": v for k, v in d.items()})
                P.adjust(b["photo"], changes or {"exposure": 0.0})
            return self._json({"ok": True})
        if path == "/api/remove":
            P.remove_reference(b["photo"])
            return self._json({"ok": True})
        if path == "/api/apply":
            if S.job.get("state") == "running":
                return self._json({"error": "already running"}, 409)
            out = str(Path(b.get("out") or P.root / "export").resolve())
            S.job = {"state": "running", "out": out, "stage": "starting", "i": 0, "n": len(P.album.photos)}
            threading.Thread(target=_apply_job, args=(S, out, b.get("max_side")), daemon=True).start()
            return self._json(S.job)
        self._json({"error": "not found"}, 404)


def _apply_job(S: State, out: str, max_side):
    from .export import export_album, review_page
    from .propagate import make_reference, propagate

    try:
        P = S.project
        refs = P.references()
        if not refs:
            raise RuntimeError("add at least one reference first")
        R = [make_reference(P.album, r.name, r.final, P.scene_of(r.name)) for r in refs]

        def prog(stage):
            return lambda i, n, name: S.job.update(stage=stage, i=i, n=n, photo=name)

        recipes = propagate(P.album, R, P.scenes, progress=prog("propagate"))
        export_album(P.album, recipes, out, int(max_side) if max_side else None, progress=prog("export"))
        review_page(P.album, P.scenes, recipes, [r.name for r in refs], out)
        S.job.update(state="done", review="/out/index.html")
    except Exception as e:  # noqa: BLE001
        S.job.update(state="error", error=str(e))


def serve(root: str | Path, host: str = "127.0.0.1", port: int = 8642) -> None:
    Handler.S = State(root)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Morph Studio on http://{host}:{port}  (project {root})")
    srv.serve_forever()
