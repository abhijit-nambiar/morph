import cv2
import numpy as np
import pytest
from PIL import Image

from dngwriter import write_dng
from morph.export import output_name
from morph.imageio import list_images, load_rgb, load_rgb_bytes, read_info, save_image


def gradient(h=60, w=90):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx / (w - 1), yy / (h - 1), np.full((h, w), 0.5)], -1).astype(np.float32)


def u8(img):
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8))


@pytest.mark.parametrize("name,save", [
    ("a.jpg", lambda im, p: im.save(p, quality=95)),
    ("b.JPEG", lambda im, p: im.save(p, "JPEG", quality=95)),
    ("c.png", lambda im, p: im.save(p)),
    ("d.png", lambda im, p: im.convert("RGBA").save(p)),
    ("e.png", lambda im, p: im.convert("P").save(p)),
    ("f.tif", lambda im, p: im.save(p)),
    ("g.webp", lambda im, p: im.save(p, quality=95)),
    ("h.bmp", lambda im, p: im.save(p)),
    ("i.avif", lambda im, p: im.save(p, quality=95)),
    ("j.jpg", lambda im, p: im.convert("CMYK").save(p, quality=95)),
])
def test_raster_formats_load(tmp_path, name, save):
    save(u8(gradient()), tmp_path / name)
    assert [p.name for p in list_images(tmp_path)] == [name]
    img = load_rgb(tmp_path / name)
    assert img.shape == (60, 90, 3) and img.dtype == np.float32
    tol = 0.08 if name.startswith("e") else 0.04  # palette quantisation
    assert np.abs(img - gradient()).mean() < tol
    assert (read_info(tmp_path / name, with_hash=False).width, read_info(tmp_path / name, with_hash=False).height) == (90, 60)


def test_heic(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    u8(gradient()).save(tmp_path / "IMG_0001.HEIC", quality=95)
    assert [p.name for p in list_images(tmp_path)] == ["IMG_0001.HEIC"]
    assert np.abs(load_rgb(tmp_path / "IMG_0001.HEIC") - gradient()).mean() < 0.04
    assert output_name("IMG_0001.HEIC") == "IMG_0001.jpg"


@pytest.mark.parametrize("name", ["rgb16.png", "rgb16.tif", "gray16.png"])
def test_16_bit_keeps_precision(tmp_path, name):
    g = gradient()
    if name.startswith("gray"):
        g = np.repeat(g[..., :1] * 0.5 + 0.25, 3, -1)
        cv2.imwrite(str(tmp_path / name), (g[..., 0] * 65535).astype(np.uint16))
    else:
        cv2.imwrite(str(tmp_path / name), (g[..., ::-1] * 65535).astype(np.uint16))
    img = load_rgb(tmp_path / name)
    assert img.shape == (60, 90, 3)
    assert np.abs(img - g).max() < 1e-4  # 8-bit truncation would leave errors up to 1/255
    assert load_rgb(tmp_path / name, max_side=45).shape == (30, 45, 3)


def test_16_bit_orientation(tmp_path):
    g = (gradient()[..., 0] * 65535).astype(np.uint16)
    exif = Image.Exif()
    exif[0x0112] = 6  # display rotated 90 degrees clockwise
    Image.fromarray(g).save(tmp_path / "x.png", exif=exif.tobytes())
    img = load_rgb(tmp_path / "x.png")
    assert img.shape == (90, 60, 3)
    assert np.abs(img[..., 0] - np.rot90(g, -1) / 65535).max() < 1e-4


def test_16_bit_rgb_orientation(tmp_path, monkeypatch):
    """OpenCV must not rotate on its own: orientation comes from Pillow's tag, applied once."""
    import cv2

    from morph import imageio

    g = gradient()
    cv2.imwrite(str(tmp_path / "x.png"), (g[..., ::-1] * 65535).astype(np.uint16))
    real = Image.Image.getexif

    def oriented(self):
        e = real(self)
        e[0x0112] = 6
        return e

    monkeypatch.setattr(Image.Image, "getexif", oriented)
    img = imageio.load_rgb(tmp_path / "x.png")
    assert np.abs(img - np.rot90(g, -1)).max() < 1e-4


def test_output_formats():
    assert output_name("a.webp") == "a.webp"
    assert output_name("a.png") == "a.png"
    assert output_name("a.bmp") == "a.jpg"
    assert output_name("a.avif") == "a.jpg"


def test_upload_bytes(tmp_path):
    g = gradient()
    cv2.imwrite(str(tmp_path / "e.tif"), (g[..., ::-1] * 65535).astype(np.uint16))
    img = load_rgb_bytes((tmp_path / "e.tif").read_bytes(), "Lightroom export.tif")
    assert np.abs(img - g).max() < 1e-4


def test_cull_judges_raw_exposure_on_the_raw(tmp_path):
    """A camera JPEG that blew the highlights must not get the RAW rejected as overexposed."""
    from morph.cull import score_folder

    yy, xx = np.mgrid[0:120, 0:180]
    lin = np.stack([0.05 + 0.25 * xx / 180, np.full((120, 180), 0.12), 0.05 + 0.25 * yy / 120], -1).astype(np.float32)
    raw_dir, jpg_dir = tmp_path / "raw", tmp_path / "jpg"
    raw_dir.mkdir()
    jpg_dir.mkdir()
    write_dng(raw_dir / "P.NEF", lin)
    hot = u8(np.clip(lin * 6, 0, 1) ** (1 / 2.2))  # what an over-bright camera rendering looks like
    hot.save(raw_dir / "P.JPG", quality=95)
    hot.save(jpg_dir / "P.JPG", quality=95)
    (r,) = score_folder(raw_dir, people=False)
    (j,) = score_folder(jpg_dir, people=False)
    assert r.scored_from.endswith("P.JPG")
    assert r.key_ev < j.key_ev - 1.0 and r.clip_hi < j.clip_hi


@pytest.mark.parametrize("src_name,dst_name", [("s.tif", "o.tif"), ("s.tif", "o.jpg"), ("s.tif", "o.png"), ("s.jpg", "o.tif")])
def test_save_keeps_metadata_without_copying_tiff_layout(tmp_path, src_name, dst_name):
    from morph.imageio import save_image

    src = tmp_path / src_name
    if src.suffix == ".tif":
        cv2.imwrite(str(src), (gradient(80, 120)[..., ::-1] * 65535).astype(np.uint16), [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    else:
        exif = Image.Exif()
        exif[0x010F] = "NIKON CORPORATION"
        u8(gradient(80, 120)).save(src, quality=95, exif=exif.tobytes())
    img = gradient(80, 120)
    out = save_image(img, tmp_path / dst_name, source=src)
    back = load_rgb(out)
    assert back.shape == img.shape
    assert np.abs(back - img).max() < 0.03


def test_jpeg_save_survives_high_entropy(tmp_path):
    noise = np.random.default_rng(0).random((3000, 4000, 3)).astype(np.float32)
    out = save_image(noise, tmp_path / "noise.jpg")
    assert Image.open(out).size == (4000, 3000)


def test_jpeg_max_bytes_keeps_resolution(tmp_path):
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.random((1200, 1800, 3)).astype(np.float32), (0, 0), 2)
    full = save_image(img, tmp_path / "full.jpg").stat().st_size
    capped = save_image(img, tmp_path / "cap.jpg", max_bytes=full // 3)
    assert capped.stat().st_size <= full // 3 and Image.open(capped).size == (1800, 1200)
    tiny = save_image(img, tmp_path / "tiny.jpg", max_bytes=1000)  # unreachable: smallest file, same size
    assert Image.open(tiny).size == (1800, 1200) and tiny.stat().st_size < capped.stat().st_size
