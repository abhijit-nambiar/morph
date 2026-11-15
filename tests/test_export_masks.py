from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from morph.export import full_res_masks
from morph.filters import upsample_mask
from morph.recipe import Recipe


@pytest.mark.parametrize("scale", [1, 6])
def test_export_preserves_tiny_face_features_without_spreading_edits(scale):
    mask = np.zeros((32, 40), np.float32)
    mask[10, 12] = 0.9
    mask[10, 26] = 0.8
    mask[20:22, 17:23] = 0.85
    # A low-contrast guide used to erase the eyes and fragment the mouth.
    img = np.full((32*scale, 40*scale, 3), 0.5, np.float32)
    album = SimpleNamespace(masks=lambda name: SimpleNamespace(masks={"eyes": mask, "lips": mask, "face_skin": mask}))
    recipe = Recipe(local={r: {"exposure": 0.2} for r in ("eyes", "lips", "face_skin")})
    masks = full_res_masks(album, "photo", img, recipe)
    for r in ("eyes", "lips"):
        out = masks[r]
        assert out.shape == img.shape[:2]
        assert out.dtype == np.float32
        assert out[:16*scale].max() > 0.7  # the single-pixel eyes survive, not only the larger mouth
        assert cv2.connectedComponents((out > 0.5).astype(np.uint8))[0] - 1 == 3
        assert out[:5*scale].max() == 0
        assert np.any((out > 0) & (out < 1))  # retain soft probabilities rather than a binary mask
        if scale > 1:
            assert np.any((out > 0) & (out < 0.8))
    # Broad regions retain their existing edge-aware upsampling.
    np.testing.assert_array_equal(masks["face_skin"], upsample_mask(mask, img[..., 0]))


def test_no_local_edits_do_not_load_masks():
    album = SimpleNamespace()
    assert full_res_masks(album, "photo", np.zeros((10, 10, 3), np.float32), Recipe()) is None
