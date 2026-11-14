import json

import numpy as np
import pytest
from PIL import Image

from dngwriter import write_dng
from morph import imageio
from morph.export import output_name, output_names
from morph.imageio import is_raw, list_images, load_preview, load_rgb, raw_pairs, read_info, save_image


def scene(h=120, w=180):
    yy, xx = np.mgrid[0:h, 0:w]
    lin = np.stack([0.04 + 0.4 * xx / w, np.full((h, w), 0.12), 0.04 + 0.4 * yy / h], -1).astype(np.float32)
    lin[40:80, 60:120] = 0.18  # neutral grey card
    return lin


@pytest.fixture
def dng(tmp_path):
    return write_dng(tmp_path / "DSC_0001.dng", scene(), date="2024:05:01 10:00:07")


def test_raw_metadata(dng):
    assert is_raw(dng) and is_raw("x.NEF") and is_raw("x.nrw") and not is_raw("x.jpg")
    info = read_info(dng)
    assert (info.width, info.height) == (180, 120)
    assert info.captured.isoformat() == "2024-05-01T10:00:07"
    assert info.iso == 400 and info.exposure_time == pytest.approx(1 / 250, rel=1e-3)
    assert info.extra.get("raw")


def test_raw_render_uses_camera_wb_and_a_pleasant_curve(dng):
    img = load_rgb(dng)
    assert img.shape == (120, 180, 3) and img.dtype == np.float32
    card = img[50:70, 75:105].reshape(-1, 3).mean(0)
    assert np.ptp(card) < 0.02  # camera white balance neutralises the as-shot cast (0.5, 1, 0.7)
    assert 0.45 < card.mean() < 0.65  # 18% grey lands at a display mid-tone, not dark linear
    assert img.max() <= 1 and img.min() >= 0
    assert load_rgb(dng, max_side=90).shape[:2] == (60, 90)
    assert load_preview(dng, 64).shape[:2] == (43, 64)  # no embedded preview: falls back to a decode


def test_raw_orientation(tmp_path):
    p = write_dng(tmp_path / "r.dng", scene(), flip=6)  # 90 deg CW
    assert load_rgb(p).shape[:2] == (180, 120)
    info = read_info(p)
    assert (info.width, info.height) == (120, 180)


def test_pairs_policy(tmp_path, dng):
    Image.new("RGB", (60, 40)).save(tmp_path / "DSC_0001.jpg")
    Image.new("RGB", (60, 40)).save(tmp_path / "DSC_0002.JPG")
    write_dng(tmp_path / "DSC_0003.NEF", scene())  # LibRaw sniffs content; the suffix just routes it
    names = lambda pol: [p.name for p in list_images(tmp_path, pairs=pol)]
    assert names("raw") == ["DSC_0001.dng", "DSC_0002.JPG", "DSC_0003.NEF"]
    assert names("jpeg") == ["DSC_0001.jpg", "DSC_0002.JPG", "DSC_0003.NEF"]
    assert len(names("both")) == 4
    assert raw_pairs(tmp_path) == {"DSC_0001": tmp_path / "DSC_0001.jpg"}
    with pytest.raises(ValueError):
        list_images(tmp_path, pairs="nope")


@pytest.mark.parametrize("suffix", ["jpg", "tif"])
def test_export_from_raw_keeps_exif(tmp_path, dng, suffix):
    out = tmp_path / f"out.{suffix}"
    save_image(load_rgb(dng), out, source=dng)
    im = Image.open(out)
    assert im.size == (180, 120)
    assert im.getexif().get_ifd(0x8769).get(0x9003) == "2024:05:01 10:00:07"
    assert im.info.get("icc_profile")


def test_output_name():
    assert output_name("DSC_1.NEF") == "DSC_1.jpg"
    assert output_name("a.jpg") == "a.jpg"
    assert output_name("a.png", "tif") == "a.tif"
    assert output_names(["portrait.jpg", "portrait.png"], "jpg") == {"portrait.jpg": "portrait.jpg", "portrait.png": "portrait-png.jpg"}
    assert output_names(["DSC_1.NEF", "DSC_1.JPG"]) == {"DSC_1.NEF": "DSC_1-nef.jpg", "DSC_1.JPG": "DSC_1.JPG"}
    assert output_names(["a.png", "a.tiff"], "jpg") == {"a.png": "a-png.jpg", "a.tiff": "a-tiff.jpg"}
    assert output_names(["a.jpg", "b.png"]) == {"a.jpg": "a.jpg", "b.png": "b.png"}


def test_album_and_export_accept_raw(fake_models, tmp_path):
    from morph.album import Album
    from morph.export import export_album
    from morph.recipe import Recipe

    d = tmp_path / "album"
    d.mkdir()
    for i in range(2):
        write_dng(d / f"R{i}.NEF", scene() * (0.8 + 0.4 * i), date=f"2024:05:01 10:00:0{i}")
    album = Album(d, tmp_path / "cache").analyze()
    assert [p.name for p in album.photos] == ["R0.NEF", "R1.NEF"]
    rep = export_album(album, {p.name: Recipe() for p in album.photos}, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out" / "photos").iterdir()) == ["R0.jpg", "R1.jpg"]
    assert all(v["file"].endswith(".jpg") for v in rep.values())


def test_cull_cache_tracks_the_paired_jpeg(tmp_path, monkeypatch):
    import os

    import cv2

    from morph import cull
    from morph.cull import score_folder

    write_dng(tmp_path / "P.NEF", scene())
    sharp = (np.clip(scene(), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    Image.fromarray(sharp).save(tmp_path / "P.JPG", quality=95)
    (first,) = score_folder(tmp_path, people=False)
    assert first.scored_from.endswith("P.JPG")
    Image.fromarray(cv2.GaussianBlur(sharp, (0, 0), 6)).save(tmp_path / "P.JPG", quality=95)
    st = (tmp_path / "P.JPG").stat()
    os.utime(tmp_path / "P.JPG", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    calls, real = [], cull._score_pixels
    monkeypatch.setattr(cull, "_score_pixels", lambda *a, **k: calls.append(1) or real(*a, **k))
    (second,) = score_folder(tmp_path, people=False)
    assert calls and second.frame_blur_px != first.frame_blur_px
    score_folder(tmp_path, people=False)
    assert len(calls) == 1  # unchanged pair: cached
