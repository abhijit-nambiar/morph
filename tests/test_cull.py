import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from morph import cull
from morph.cli import main
from morph.cull import CullSettings, FrameScore, build_album, decide, score_folder


def texture(seed, h=600, w=900):
    """Photo-like content: shapes with crisp edges over multi-scale texture."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for s in (40, 12, 3):
        n = cv2.resize(rng.random((h // s + 1, w // s + 1, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        img += 0.18 * n
    for _ in range(25):
        c = rng.random(3) * 0.8 + 0.1
        x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(10, 90))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c.tolist(), -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(r * 0.7)), c.tolist(), -1)
    return np.clip(img, 0, 1)


def blur(img, r):
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1, -1)
    return cv2.filter2D(img, -1, k / k.sum())


def shake(img, length, angle=20):
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx, dy = np.cos(np.radians(angle)) * c, np.sin(np.radians(angle)) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1, cv2.LINE_AA)
    return cv2.filter2D(img, -1, k / k.sum())


def save(img, path, second):
    exif = Image.Exif()
    stamp = f"2024:05:01 10:{second // 60:02d}:{second % 60:02d}"
    exif.get_ifd(0x8769)[0x9003] = stamp
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=93, exif=exif.tobytes())


@pytest.fixture
def dump(tmp_path):
    d = tmp_path / "dump"
    d.mkdir()
    a, b, c = texture(1), texture(2), texture(3)
    shifted = np.roll(a, (4, 6), (0, 1)) * 0.97
    save(a, d / "001.jpg", 0)
    save(shifted, d / "002.jpg", 1)  # near-duplicate of 001
    save(blur(a, 5), d / "003.jpg", 2)  # missed focus, same burst
    save(b, d / "004.jpg", 60)
    save(shake(b, 21), d / "005.jpg", 61)  # camera shake
    save(c, d / "006.jpg", 120)
    save(c**2.2 * 0.08, d / "007.jpg", 121)  # ~3.5 EV underexposed (ok to group with 006 or not)
    save(np.clip(texture(4) * 4, 0, 1), d / "008.jpg", 180)  # blown out
    return d


def test_blur_estimate_tracks_defocus():
    img = texture(5)
    g = lambda im: cull._Grad(cull._gray(im)).sigma()
    s0, s2, s4, s7 = g(img), g(blur(img, 2)), g(blur(img, 4)), g(blur(img, 7))
    assert s0 < s2 < s4 < s7
    assert s0 < 0.8 and s4 > 1.4
    # contrast-independent: a flat, low-contrast copy measures the same blur
    assert abs(g(0.5 + 0.2 * (img - 0.5)) - s0) < 0.15


def test_motion_is_directional():
    img = texture(6)
    m0, _ = cull._Grad(cull._gray(img)).motion()
    m1, ang = cull._Grad(cull._gray(shake(img, 21, 0))).motion()
    mb, _ = cull._Grad(cull._gray(blur(img, 5))).motion()
    assert m0 < 0.15 and mb < 0.15 and m1 > 0.3
    assert ang in (0, 180)


def test_score_decide_and_cache(dump, monkeypatch):
    scores = score_folder(dump, people=False)
    assert [s.name for s in scores] == [f"00{i}.jpg" for i in range(1, 9)]
    json.dumps([s.to_json() for s in scores])
    by = {s.name: s for s in scores}
    assert by["001.jpg"].burst_id == by["002.jpg"].burst_id == by["003.jpg"].burst_id
    assert by["004.jpg"].burst_id != by["001.jpg"].burst_id
    assert by["003.jpg"].focus_blur_px > by["001.jpg"].focus_blur_px + 0.8
    assert by["005.jpg"].motion > 0.3
    assert by["007.jpg"].key_ev < -2.5 and by["008.jpg"].subject_clip_hi > 0.2

    dec = {d.name: d for d in decide(scores)}
    keep = sorted(n for n, d in dec.items() if d.keep)
    assert keep == ["004.jpg", "006.jpg"] + [min(("001.jpg", "002.jpg"), key=lambda n: -by[n].quality)] or keep == sorted(
        ["004.jpg", "006.jpg", max(("001.jpg", "002.jpg"), key=lambda n: by[n].quality)])
    assert "soft" in dec["003.jpg"].flags and "out of focus" in dec["003.jpg"].reasons[0]
    assert "motion" in dec["005.jpg"].flags
    assert "under" in dec["007.jpg"].flags
    assert "clipped" in dec["008.jpg"].flags or "over" in dec["008.jpg"].flags
    dup = dec["002.jpg"] if dec["001.jpg"].keep else dec["001.jpg"]
    assert dup.flags == ["duplicate"] and "near-duplicate" in dup.reasons[0]

    # settings: keep two per burst, lenient focus
    dec2 = {d.name: d for d in decide(scores, CullSettings(burst_keep=2))}
    assert dec2["001.jpg"].keep and dec2["002.jpg"].keep and not dec2["003.jpg"].keep
    dec3 = {d.name: d for d in decide(scores, CullSettings(exposure_tolerance_ev=7, raw_latitude_ev=0))}
    assert "under" not in dec3["007.jpg"].flags

    # re-scoring is served from the cache
    calls = []
    monkeypatch.setattr(cull, "load_preview", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError))
    again = score_folder(dump, people=False)
    assert not calls
    for x, y in zip(again, scores):
        assert json.dumps(x.to_json()) == json.dumps(y.to_json())


def test_strictness_is_monotone(dump):
    scores = score_folder(dump, people=False)
    kept = [sum(d.keep for d in decide(scores, CullSettings(focus_strictness=s, burst_keep=9))) for s in (0, 0.5, 1)]
    assert kept[0] >= kept[1] >= kept[2]
    assert CullSettings(focus_strictness=0).max_blur_px > CullSettings().max_blur_px > CullSettings(focus_strictness=1).max_blur_px


def test_eye_openness():
    assert cull.eye_openness({"box": [0, 0, 100, 100], "eyes": [{"aspect": 0.4}]}) == 1
    assert cull.eye_openness({"box": [0, 0, 100, 100], "eyes": [{"aspect": 0.08}, {"aspect": 0.09}]}) == 0
    assert cull.eye_openness({"box": [0, 0, 400, 400], "eyes": [], "brows": 2}) < 0.25  # lids shut
    assert cull.eye_openness({"box": [0, 0, 200, 200], "eyes": [], "brows": 2}) is None  # too small to infer a blink
    assert cull.eye_openness({"box": [0, 0, 50, 50], "eyes": [{"aspect": 0.08}]}) is None  # tiny background face
    assert cull.eye_openness({"box": [0, 0, 100, 100], "eyes": [], "brows": 0}) is None  # back of head / profile


def _fake(name, burst, q, flags=(), **kw):
    base = dict(name=name, path=f"/x/{name}", scored_from=f"/x/{name}", captured=None, width=10, height=10,
                focus_region="frame", focus_blur_px=0.5, frame_blur_px=0.5, subject_blur_px=None, eye_blur_px=None,
                motion=0.0, motion_angle=0.0, key_ev=0.0, clip_hi=0.0, clip_lo=0.0, subject_clip_hi=0.0)
    base.update(kw)
    return FrameScore(**base)


def test_closed_eyes_and_keep_one():
    s = [_fake("a.jpg", 0, 1, eyes_closed=2, eyes_open=1, descriptor=[1.0] * 780),
         _fake("b.jpg", 0, 1, eyes_closed=0, eyes_open=3, descriptor=[1.0] * 780)]
    d = {x.name: x for x in decide(s)}
    assert not d["a.jpg"].keep and "eyes_closed" in d["a.jpg"].flags and d["b.jpg"].keep
    d = {x.name: x for x in decide(s, CullSettings(reject_closed_eyes=False, burst_keep=2))}
    assert d["a.jpg"].keep
    # a scene with nothing but slightly soft frames keeps its best one
    soft = [_fake("c.jpg", 0, 1, focus_blur_px=1.5, descriptor=[1.0] * 780), _fake("d.jpg", 0, 1, focus_blur_px=1.6, descriptor=[1.0] * 780)]
    very_soft = [_fake("e.jpg", 0, 1, focus_blur_px=3.0)]
    d = {x.name: x for x in decide(soft)}
    assert d["c.jpg"].keep and not d["d.jpg"].keep and "best available" in d["c.jpg"].reasons[0]
    assert not any(x.keep for x in decide(soft, CullSettings(keep_at_least_one=False)))
    assert not decide(very_soft)[0].keep


def test_build_album(dump, tmp_path):
    from morph.imageio import list_images

    dec = decide(score_folder(dump, people=False))
    kept = sorted(d.name for d in dec if d.keep)
    out = tmp_path / "album"
    build_album(dec, out)
    assert sorted(p.name for p in list_images(out)) == kept
    assert all((out / n).is_symlink() for n in kept)
    # overrides by name or stem; files placed earlier but now rejected are removed
    build_album(dec, out, overrides={"005": "keep", kept[0]: False})
    assert sorted(p.name for p in list_images(out)) == sorted(set(kept) - {kept[0]} | {"005.jpg"})
    copy = tmp_path / "copy"
    build_album(dec, copy, mode="copy")
    assert not any(p.is_symlink() for p in copy.iterdir())
    (tmp_path / "foreign").mkdir()
    (tmp_path / "foreign" / kept[0]).write_bytes(b"mine")
    with pytest.raises(FileExistsError):
        build_album(dec, tmp_path / "foreign")


def test_cli(dump, tmp_path, capsys):
    main(["cull", str(dump), "--dry-run", "--no-people"])
    out = capsys.readouterr().out
    assert "8 frames" in out and "keep 3" in out and not (tmp_path / "album").exists()
    main(["cull", str(dump), "--out", str(tmp_path / "album"), "--no-people", "--burst-keep", "2", "--reject", "004.jpg"])
    rep = json.loads((tmp_path / "album" / "cull.json").read_text())
    assert rep["summary"]["frames"] == 8 and rep["settings"]["burst_keep"] == 2
    assert {f["name"] for f in rep["frames"]} == {f"00{i}.jpg" for i in range(1, 9)}
    names = {p.name for p in (tmp_path / "album").iterdir() if p.suffix == ".jpg"}
    assert names == {"001.jpg", "002.jpg", "006.jpg"}


def test_overrides_schema_and_scores(dump):
    scores = score_folder(dump, people=False)
    for s in scores:
        assert all(0 <= v <= 1 for v in (s.focus, s.exposure, s.stability, s.quality)) and s.reasons
        assert {"focus", "exposure", "stability", "quality", "reasons", "burst_id", "burst_rank", "burst_size", "keep"} <= set(s.to_json())
    by = {s.name: s for s in scores}
    assert by["005.jpg"].stability < 0.5 and by["007.jpg"].exposure < 0.5 and by["003.jpg"].focus < 0.5
    dec = {d.name: d for d in decide(scores, overrides={"005.jpg": True, "004": False, "006.jpg": None})}
    assert dec["005.jpg"].keep and dec["005.jpg"].override is True and dec["005.jpg"].auto_keep is False
    assert "manual keep" in dec["005.jpg"].reasons[0]
    assert not dec["004.jpg"].keep and dec["004.jpg"].override is False
    assert dec["006.jpg"].override is None and dec["006.jpg"].keep
    sch = CullSettings.schema()
    keys = {r["key"] for r in sch}
    assert keys == set(CullSettings().to_json())
    for r in sch:
        assert r["type"] in ("float", "int", "bool") and r["label"] and r["help"]
        if r["type"] != "bool":
            assert r["min"] <= r["default"] <= r["max"]
    json.dumps(sch)


def test_decode_ahead_is_bounded(tmp_path, monkeypatch):
    import threading
    for i in range(24):
        save(texture(i, 120, 160), tmp_path / f"{i:03d}.jpg", i * 10)
    live, peak, lock = [0], [0], threading.Lock()
    real_load, real_score = cull.load_preview, cull._score_pixels

    def load(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        return real_load(*a, **k)

    def score(*a, **k):
        with lock:
            live[0] -= 1
        return real_score(*a, **k)

    monkeypatch.setattr(cull, "load_preview", load)
    monkeypatch.setattr(cull, "_score_pixels", score)
    assert len(score_folder(tmp_path, people=False, workers=2)) == 24
    assert peak[0] <= 2 * 2 + 1


def test_zero_thresholds_do_not_crash(dump):
    assert cull._ramp(0.0, 0.0, 0.0) == 1 and cull._ramp(0.1, 0.0, 0.0) == 0
    scores = score_folder(dump, people=False)
    lo = {r["key"]: r["min"] for r in CullSettings.schema() if r["type"] != "bool"}
    for d in decide(scores, CullSettings(**lo)):
        assert 0 <= d.quality <= 1 and 0 <= d.exposure <= 1


def test_cache_follows_a_moved_dump(dump, tmp_path, monkeypatch):
    import shutil
    first = {s.name: s for s in score_folder(dump, people=False)}
    moved = shutil.move(str(dump), str(tmp_path / "moved"))
    monkeypatch.setattr(cull, "_score_pixels", lambda *a, **k: pytest.fail("cache miss after move"))
    for s in score_folder(moved, people=False):
        assert s.path == str((Path(moved) / s.name).resolve()) and Path(s.path).exists()
        assert Path(s.scored_from).exists() and Path(s.preview).exists()
        assert s.preview.startswith(str(Path(moved).resolve())) and s.focus_blur_px == first[s.name].focus_blur_px


def test_build_album_switches_link_and_copy(dump, tmp_path):
    decisions = decide(score_folder(dump, people=False))
    album = tmp_path / "album"
    linked = build_album(decisions, album, mode="link")
    assert linked and all(p.is_symlink() for p in linked)
    copied = build_album(decisions, album, mode="copy")
    assert [p.name for p in copied] == [p.name for p in linked]
    assert all(p.is_file() and not p.is_symlink() for p in copied)
    mtimes = {p.name: p.stat().st_mtime_ns for p in copied}
    again = build_album(decisions, album, mode="copy")  # unchanged copies are left alone
    assert {p.name: p.stat().st_mtime_ns for p in again} == mtimes
    assert all(p.is_symlink() for p in build_album(decisions, album, mode="link"))


def test_failed_frame_does_not_split_a_burst():
    rng = np.random.default_rng(0)
    a = rng.normal(size=32 * 24)
    a /= np.linalg.norm(a)
    other = rng.normal(size=32 * 24)
    other /= np.linalg.norm(other)
    col = [0.4] * 36
    frames = [_fake(f"{i}.jpg", 0, 0, descriptor=list(v) + c, captured=f"2026-01-01T10:00:0{i}")
              for i, (v, c) in enumerate([(a, col), (a, [0.02] * 36), (a * 0.99 + other * 0.05, col), (other, col)])]
    cull.group_bursts(frames, CullSettings())
    ids = [f.burst_id for f in frames]
    assert ids[0] == ids[2] != ids[1] and ids[3] not in (ids[0], ids[1])


def test_build_album_refuses_to_cull_in_place(dump):
    decisions = decide(score_folder(dump, people=False))
    before = {f.name: f.read_bytes() for f in dump.iterdir() if f.is_file()}
    with pytest.raises(ValueError):
        build_album(decisions, dump, mode="link")
    assert all((dump / n).read_bytes() == b and not (dump / n).is_symlink() for n, b in before.items())
