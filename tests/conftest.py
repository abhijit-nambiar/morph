import torch

torch.set_num_threads(2)

import numpy as np
import pytest
from PIL import Image

from morph import album as album_mod
from morph.segment import REGIONS, RegionMasks


class FakeSegmenter:
    """Geometric stand-in for the neural parsers: a 'person' ellipse with a face, hair and clothes."""

    def segment(self, u8):
        h, w = u8.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        person = (((xx - w / 2) / (w * 0.22)) ** 2 + ((yy - h * 0.6) / (h * 0.42)) ** 2 < 1).astype(np.float32)
        face = (((xx - w / 2) / (w * 0.08)) ** 2 + ((yy - h * 0.32) / (h * 0.09)) ** 2 < 1).astype(np.float32)
        hair = (((xx - w / 2) / (w * 0.1)) ** 2 + ((yy - h * 0.24) / (h * 0.06)) ** 2 < 1).astype(np.float32) * (1 - face)
        clothes = person * (yy > h * 0.45)
        m = {r: np.zeros((h, w), np.float32) for r in REGIONS}
        m.update(subject=person, background=1 - person, face_skin=face, hair=hair, clothes=clothes,
                 body_skin=np.clip(person - clothes - face - hair, 0, 1) * 0.5, sky=(1 - person) * (yy < h * 0.2))
        return RegionMasks(m, [{"box": [0, 0, 1, 1], "area_frac": 0.02}], [{"box": [0, 0, 1, 1], "area_frac": 0.3}])


class FakeEmbedder:
    def __call__(self, img, background):
        v = np.concatenate([img.mean((0, 1)), img[: img.shape[0] // 4].mean((0, 1)), [1.0]]).astype(np.float32)
        v = v - 0.4
        return v / np.linalg.norm(v), v / np.linalg.norm(v)


def make_photo(path, seed, tint=(1.0, 1.0, 1.0), gain=1.0, size=(240, 160)):
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([0.3 + 0.3 * xx / w, 0.5 - 0.2 * yy / h, 0.4 + 0.1 * np.sin(xx / 7)], -1)
    person = (((xx - w / 2) / (w * 0.22)) ** 2 + ((yy - h * 0.6) / (h * 0.42)) ** 2 < 1)[..., None]
    img = np.where(person, np.array([0.75, 0.55, 0.45]), bg) * np.array(tint) * gain
    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)
    exif = Image.Exif()
    exif[0x0112] = 1
    exif[0x010F] = "TestCam"
    Image.fromarray((img * 255).astype(np.uint8)).save(path, quality=95, exif=exif.tobytes())
    return path


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(album_mod, "Segmenter", FakeSegmenter)
    monkeypatch.setattr(album_mod, "Embedder", FakeEmbedder)


@pytest.fixture
def tiny_album(tmp_path):
    d = tmp_path / "album"
    d.mkdir()
    for i in range(4):
        make_photo(d / f"a{i}.jpg", i, tint=(1.0, 1.0, 1.0), gain=0.85 + 0.1 * i)
    for i in range(3):
        make_photo(d / f"b{i}.jpg", 10 + i, tint=(0.7, 0.9, 1.25), gain=0.9 + 0.1 * i)
    return d
