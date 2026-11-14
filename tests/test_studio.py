import functools
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from morph import cull, studio
from tests.test_cull import dump  # noqa: F401  (fixture)


@pytest.fixture
def server(tmp_path, dump, fake_models, monkeypatch):  # noqa: F811
    monkeypatch.setattr(cull, "score_folder", functools.partial(cull.score_folder, people=False))
    studio.Handler.S = studio.State(tmp_path / "proj", dump)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), studio.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data, {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                return r.status, json.loads(raw) if r.headers["Content-Type"] == "application/json" else raw
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(kind):
        for _ in range(600):
            j = call("/api/job")[1]
            if j.get("kind") == kind and j["state"] != "running":
                assert j["state"] == "done", j
                return j
            time.sleep(0.1)
        raise TimeoutError(kind)

    yield call, wait
    srv.shutdown()


def test_cull_to_album_flow(server, tmp_path):
    call, wait = server
    assert call("/api/state")[1]["scenes"] is None
    assert call("/api/cull")[1]["state"] == "scoring"
    wait("score")
    c = call("/api/cull")[1]
    assert c["state"] == "ready" and len(c["frames"]) == 8 and c["schema"]
    kept = {f["name"] for f in c["frames"] if f["keep"]}
    assert len(kept) == 3 and all(f["reasons"] for f in c["frames"])
    assert call("/img/cull/001.jpg")[1][:2] == b"\xff\xd8"

    c = call("/api/cull/decide", {"settings": {**c["settings"], "burst_keep": 2}})[1]
    assert len([f for f in c["frames"] if f["keep"]]) == 4
    c = call("/api/cull/override", {"name": "004.jpg", "keep": False})[1]
    assert {f["name"]: f["override"] for f in c["frames"]}["004.jpg"] is False
    sess = json.loads((tmp_path / "proj" / "studio.json").read_text())
    assert sess["overrides"] == {"004.jpg": False} and sess["cull_settings"]["burst_keep"] == 2

    album = tmp_path / "proj" / "album"
    assert call("/api/cull/build", {"out": str(album), "mode": "copy"})[1]["kept"] == 3
    wait("analyse")
    st = call("/api/state")[1]
    assert set(st["photos"]) == {p.name for p in album.iterdir() if p.suffix == ".jpg"} and len(st["photos"]) == 3

    call("/api/cull/override", {"name": "004.jpg", "keep": None})
    assert call("/api/cull/build", {"out": str(album), "mode": "copy"})[1]["kept"] == 4  # rebuild in place
    wait("analyse")
    assert len(call("/api/state")[1]["photos"]) == 4

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "001.jpg").write_bytes(b"mine")
    code, r = call("/api/cull/build", {"out": str(foreign)})
    assert code == 409 and "not created by morph cull" in r["error"] and (foreign / "001.jpg").read_bytes() == b"mine"


def test_security(server):
    call, _ = server
    assert call("/api/state", headers={"Host": "evil.example"})[0] == 403
    assert call("/api/state", headers={"Origin": "http://evil.example"})[0] == 403
    assert call("/api/start", {"folder": "/"}, headers={"Content-Type": "text/plain"})[0] == 403
    assert call("/api/generate", {"photo": "x", "prompt": "warm"})[1]["error"]
    code, body = call("/api/start", {"folder": "/nonexistent-folder"})
    assert code == 422 and body["error"].startswith("Not a folder") and "trace" not in body
