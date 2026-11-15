import cv2
import numpy as np
import pytest

from morph import segment
from morph.filters import guided_filter


def blob_count(mask):
    return cv2.connectedComponents((mask > 0.5).astype(np.uint8), connectivity=8)[0] - 1


@pytest.mark.parametrize("size", [1, 2, 4])
def test_feature_mask_preserves_small_features(size):
    mask = np.zeros((64, 64), np.float32)
    mask[20:20 + size, 16:16 + size] = 0.95
    mask[20:20 + size, 32:32 + size] = 0.85
    mask[35:35 + size, 22:28] = 0.9
    result = segment._feature_mask(mask, np.ones_like(mask))
    np.testing.assert_array_equal(result, mask)
    assert result.max() == mask.max()
    assert blob_count(result) == 3
    assert np.count_nonzero(result[mask == 0]) == 0


def test_feature_mask_empty_and_person_gate():
    empty = np.zeros((24, 24), np.float32)
    np.testing.assert_array_equal(segment._feature_mask(empty, np.ones_like(empty)), empty)
    np.testing.assert_array_equal(segment._feature_mask(np.ones_like(empty), empty), empty)
    mask = np.full_like(empty, 0.8)
    person = np.full_like(empty, 0.4)
    np.testing.assert_array_equal(segment._feature_mask(mask, person), mask * np.clip(person * 1.15, 0, 1))


def test_segment_preserves_other_regions(monkeypatch):
    rng = np.random.default_rng(7)
    h, w = 80, 96
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    human = {k: rng.uniform(0, 0.2, (h, w)).astype(np.float32) for k in segment._HUMAN_GROUPS}
    human["background"][:] = 0.1
    scene = {"sky": rng.random((h, w), dtype=np.float32),
             "vegetation": rng.random((h, w), dtype=np.float32)}
    maps = {k: rng.random((h, w), dtype=np.float32) for k in ("hair", "face_skin", "body_skin", "clothes")}
    maps["eyes"] = np.zeros((h, w), np.float32)
    maps["lips"] = np.zeros((h, w), np.float32)
    maps["eyes"][24:26, 30:33] = 0.95
    maps["eyes"][24:26, 44:47] = 0.9
    maps["lips"][39:41, 34:44] = 0.92
    model = segment.Segmenter()
    monkeypatch.setattr(model, "_refine_people", lambda full, small, parts, scale: parts)
    monkeypatch.setattr(segment, "_gate_people", lambda parts, person: parts)
    def groups(probs, labels, names):
        return human if names is segment._HUMAN_GROUPS else scene
    monkeypatch.setattr(segment, "_group_probs", groups)
    monkeypatch.setattr(model, "_probs", lambda *a, **kw: (np.ones((1, h, w), np.float32), {0: "person"}))
    def parse(full, small, parts, person, target, scale):
        target.update({k: v.copy() for k, v in maps.items()})
        return []
    monkeypatch.setattr(model, "_parse_faces", parse)
    result = model.segment(img)
    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    def snap(m):
        return np.clip(guided_filter(guide, np.clip(m, 0, 1), 2, 1e-3), 0, 1).astype(np.float32)
    person = snap(1 - human["background"])
    background = np.clip(1 - person, 0, 1)
    expected = {"subject": person, "background": background,
                "sky": snap(scene["sky"]) * background,
                "vegetation": snap(scene["vegetation"]) * background}
    for k in ("hair", "face_skin", "body_skin", "clothes"):
        expected[k] = snap(maps[k]) * np.clip(person * 1.15, 0, 1)
    for k, value in expected.items():
        np.testing.assert_array_equal(result.masks[k], value)
    for k in ("eyes", "lips"):
        np.testing.assert_array_equal(result.masks[k], segment._feature_mask(maps[k], person))
        assert blob_count(result.masks[k]) == blob_count(maps[k])
        assert result.masks[k].max() == maps[k].max()
