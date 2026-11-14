"""Morph Studio: a small local web app for culling a camera dump, building reference edits and running the album.

Flow: (optionally) point it at a camera dump, score/cull frames and build the base album -> pick a suggested photo -> prompt a generator for candidates (each checked by the distortion
guard) or upload your own edit -> pick one -> fine-tune global / per-region sliders with a live
preview -> repeat for a few photos -> apply to the whole album and open the review page.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import threading
import traceback
from dataclasses import asdict
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
    def __init__(self, root: str | Path, dump: str | Path | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project: Project | None = None
        self.lock = threading.Lock()
        self.job: dict = {"state": "idle"}
        self._suggest = None
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}
        self._session = self.root / "studio.json"
        sess = json.loads(self._session.read_text()) if self._session.exists() else {}
        self.dump: str | None = str(Path(dump).resolve()) if dump else sess.get("dump")
        self.cull_settings: dict | None = sess.get("cull_settings")
        self.overrides: dict[str, bool] = sess.get("overrides", {})
        self.scores = None
        self._thumbs: dict[tuple[str, int], bytes] = {}
        self.reload_project()
        if self.dump and dump:
            self.save_session()

    def reload_project(self):
        self.project = Project(self.root) if (self.root / "project.json").exists() else None
        self._suggest = None
        self._cache.clear()

    def save_session(self):
        self._session.write_text(json.dumps({"dump": self.dump, "cull_settings": self.cull_settings, "overrides": self.overrides}, indent=1))

    def run(self, kind: str, fn, n: int = 0, **extra) -> dict:
        """Run fn(progress) in the background as the single Studio job."""
        if self.job.get("state") == "running":
            raise RuntimeError("another job is still running")
        self.job = {"state": "running", "kind": kind, "stage": kind, "i": 0, "n": n, **extra}

        def progress(i, n, name=""):
            self.job.update(i=i, n=n, photo=Path(str(name)).name)

        def target():
            try:
                fn(progress)
                self.job.update(state="done")
            except Exception as e:  # noqa: BLE001
                self.job.update(state="error", error=str(e))

        threading.Thread(target=target, daemon=True).start()
        return self.job

    # ---- culling -------------------------------------------------------------------------------
    def start_scoring(self) -> dict:
        from .cull import score_folder

        def fn(progress):
            self.scores = score_folder(self.dump, cache=self.root / "cull-cache", progress=progress)

        self.scores = None
        return self.run("score", fn)

    def settings_obj(self):
        from .cull import CullSettings

        if not self.cull_settings:
            return CullSettings()
        known = CullSettings.__dataclass_fields__
        return CullSettings(**{k: v for k, v in self.cull_settings.items() if k in known})

    def decisions(self):
        from .cull import decide

        return decide(self.scores, self.settings_obj(), overrides=self.overrides)

    def frame_path(self, name: str) -> Path:
        for sc in self.scores or []:
            if sc.name == name:
                return Path(sc.path)
        raise KeyError(name)

    def cull_state(self) -> dict:
        if not self.dump:
            return {"state": "none"}
        from .cull import CullSettings

        if self.scores is None:
            if self.job.get("kind") != "score" or self.job.get("state") != "running":
                self.start_scoring()
            return {"state": "scoring", "job": self.job}
        decs = {d.name: d for d in self.decisions()}
        frames = []
        for sc in self.scores:
            d = decs[sc.name]
            frames.append({**sc.to_json(), "keep": bool(d.keep), "override": d.override, "reasons": list(d.reasons)})
        kept = sum(f["keep"] for f in frames)
        bursts = len({f.get("burst_id") for f in frames if f.get("burst_size", 1) > 1})
        auto_rej = sum(1 for f in frames if not f["keep"] and f["override"] is None)
        summary = [["Frames", str(len(frames))], ["Kept", str(kept)], ["Rejected automatically", str(auto_rej)],
                   ["Manual picks", str(sum(f["override"] is not None for f in frames))], ["Bursts", str(bursts)]]
        return {"state": "ready", "dump": self.dump, "frames": frames, "settings": asdict(self.settings_obj()),
                "schema": CullSettings.schema(), "summary": summary, "album_default": str(self.root / "album")}

    def build(self, out: str, mode: str) -> dict:
        from .cull import build_album

        decs = self.decisions()
        out_dir = Path(out).expanduser().resolve()
        build_album(decs, out_dir, mode=mode)
        kept = sum(bool(d.keep) for d in decs)
        return {"kept": kept, "job": self.start_analysis(out_dir)}

    # ---- album -----------------------------------------------------------------------------------
    def start_analysis(self, album_dir: str | Path) -> dict:
        def fn(progress):
            Project.create(self.root, album_dir, progress=progress)
            self.reload_project()

        return self.run("analyse", fn)

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
        base = {"project": self.root.name, "cull": bool(self.dump), "job": self.job, "export_default": str(self.root / "export")}
        if p is None:
            return {**base, "scenes": None}
        refs = {r.name: r for r in p.references()}
        return {
            **base,
            "scenes": [s.to_json() for s in p.scenes],
            "photos": {ph.name: {"faces": len(ph.faces), "scene": p.scene_of(ph.name), "size": [ph.width, ph.height]} for ph in p.album.photos},
            "suggestions": [s.__dict__ for s in self.suggestions()],
            "references": {n: {"source": r.source, "prompt": r.prompt, "guard": r.guard, "fit": r.fit, "final": r.final.to_json(), "adjust": r.adjust.to_json()} for n, r in refs.items()},
            "regions": ["global"] + list(REGIONS),
            "global_keys": list(GLOBAL_KEYS),
            "local_keys": list(LOCAL_KEYS),
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

    def _trusted(self, post: bool) -> bool:
        """Block DNS-rebinding (Host) and cross-site requests (Origin / non-JSON POSTs)."""
        host = urlparse("//" + self.headers.get("Host", "")).hostname or ""
        if host != "localhost":  # DNS rebinding needs a domain name; plain IP Hosts (e.g. LAN access) are fine
            try:
                ipaddress.ip_address(host)
            except ValueError:
                return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host", ""):
            return False
        return not post or self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json"

    def do_GET(self):
        if not self._trusted(False):
            return self._json({"error": "forbidden"}, 403)
        try:
            self._get()
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self):
        if not self._trusted(True):
            return self._json({"error": "forbidden"}, 403)
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
        if u.path == "/api/job":
            return self._json(S.job)
        if u.path == "/api/cull":
            return self._json(S.cull_state())
        if parts[0] == "img" and len(parts) >= 3 and parts[1] in ("cull", "cullfull"):
            side = 480 if parts[1] == "cull" else 1800
            key = (parts[2], side)
            if key not in S._thumbs:
                from .imageio import load_rgb

                S._thumbs[key] = _jpeg(load_rgb(S.frame_path(parts[2]), max_side=side), side, 85)
            return self._send(S._thumbs[key], "image/jpeg")
        if parts[0] in ("img", "api") and S.project is None:
            return self._json({"error": "no album yet"}, 409)
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
            root = Path(S.job["out"]).resolve()
            f = (root / "/".join(parts[1:])).resolve()
            if f.is_file() and f.is_relative_to(root):
                ctype = {".html": "text/html", ".jpg": "image/jpeg", ".json": "application/json"}.get(f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
        self._json({"error": "not found"}, 404)

    def _post(self, b):
        S, path = self.S, urlparse(self.path).path
        try:
            r = self._post_setup(S, path, b)
        except RuntimeError as e:
            return self._json({"error": str(e)}, 409)
        if r is not None:
            return self._json(r)
        P = S.project
        if P is None:
            return self._json({"error": "no album yet"}, 409)
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
            out = str(Path(b.get("out") or P.root / "export").expanduser().resolve())
            S.job = {"state": "running", "kind": "apply", "out": out, "stage": "starting", "i": 0, "n": len(P.album.photos)}
            threading.Thread(target=_apply_job, args=(S, out, b.get("max_side")), daemon=True).start()
            return self._json(S.job)
        self._json({"error": "not found"}, 404)


    def _post_setup(self, S: State, path: str, b: dict):
        """Endpoints that work before an album exists: start, culling and album building."""
        if path == "/api/start":
            folder = Path(str(b.get("folder", ""))).expanduser().resolve()
            if not folder.is_dir():
                raise RuntimeError(f"not a folder: {folder}")
            if b.get("cull"):
                S.dump, S.scores, S.overrides = str(folder), None, {}
                S.save_session()
                return {"job": S.start_scoring()}
            return {"job": S.start_analysis(folder)}
        if path == "/api/cull/decide":
            if S.scores is None:
                raise RuntimeError("frames are still being scored")
            S.cull_settings = b.get("settings") or None
            S.save_session()
            return S.cull_state()
        if path == "/api/cull/override":
            if S.scores is None:
                raise RuntimeError("frames are still being scored")
            if b.get("keep") is None:
                S.overrides.pop(b["name"], None)
            else:
                S.overrides[b["name"]] = bool(b["keep"])
            S.save_session()
            return S.cull_state()
        if path == "/api/cull/build":
            if S.scores is None:
                raise RuntimeError("frames are still being scored")
            mode = b.get("mode", "link")
            if mode not in ("link", "copy"):
                raise RuntimeError("mode must be link or copy")
            return S.build(str(b.get("out") or S.root / "album"), mode)
        return None


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

        S.job.update(stage="propagate")
        recipes = propagate(P.album, R, P.scenes, progress=prog("propagate"))
        export_album(P.album, recipes, out, int(max_side) if max_side else None, progress=prog("export"))
        review_page(P.album, P.scenes, recipes, [r.name for r in refs], out)
        S.job.update(state="done", review="/out/index.html")
    except Exception as e:  # noqa: BLE001
        S.job.update(state="error", error=str(e))


def serve(root: str | Path, host: str = "127.0.0.1", port: int = 8642, dump: str | Path | None = None) -> None:
    Handler.S = State(root, dump)
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Morph Studio on http://{host}:{port}  (project {root})")
    srv.serve_forever()
