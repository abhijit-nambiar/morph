import numpy as np
import torch

from morph.recipe import Recipe
from morph.render import Params, render, render_recipe, to_tensor
from morph.segment import REGIONS


def _img(h=64, w=96, seed=0):
    return np.random.default_rng(seed).random((h, w, 3)).astype(np.float32)


def test_identity_is_exact():
    img = _img()
    masks = {r: np.full(img.shape[:2], 0.5, np.float32) for r in REGIONS}
    assert np.abs(render_recipe(img, masks, Recipe()) - img).max() < 1e-6


def test_shape_and_range_preserved():
    img = _img()
    r = Recipe({"exposure": 1.5, "contrast": 0.8, "grain": 0.5, "clarity": 0.7, "hsl_red_sat": 1.0, "vignette": -0.5})
    out = render_recipe(img, None, r)
    assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1


def test_local_edit_stays_in_mask():
    img = _img()
    m = np.zeros(img.shape[:2], np.float32)
    m[:, :40] = 1
    out = render_recipe(img, {"face_skin": m}, Recipe(local={"face_skin": {"exposure": 0.8, "temperature": 0.4}}))
    assert np.abs(out[:, 50:] - img[:, 50:]).max() < 1e-6
    assert np.abs(out[:, :30] - img[:, :30]).mean() > 0.02


def test_exposure_monotonic():
    img = _img()
    lo = render_recipe(img, None, Recipe({"exposure": -0.5})).mean()
    hi = render_recipe(img, None, Recipe({"exposure": 0.5})).mean()
    assert lo < img.mean() < hi


def test_differentiable():
    img = _img()
    p = Params.from_recipe(Recipe(), requires_grad=True)
    masks = torch.rand(1, len(REGIONS), 64, 96)
    render(to_tensor(img), masks, p).mean().backward()
    assert p.gv.grad.abs().sum() > 0 and p.lv.grad.abs().sum() > 0


def test_recipe_json_roundtrip(tmp_path):
    r = Recipe({"exposure": 0.3, "hsl_blue_sat": -0.2}, {"hair": {"saturation": 0.1}}, {"x": 1})
    r.save(tmp_path / "r.json")
    q = Recipe.load(tmp_path / "r.json")
    assert q.global_ == r.global_ and q.local == r.local


def test_adjustment_delta_beyond_slider_range_survives_save():
    from morph.project import ReferenceEdit
    from morph.recipe import Recipe

    ref = ReferenceEdit("a.jpg", Recipe({"contrast": 0.8}), Recipe({"contrast": -1.8}))
    back = ReferenceEdit.from_json(ref.to_json())
    assert back.adjust.g("contrast") == -1.8 and back.final.g("contrast") == -1.0
