import torch

torch.set_num_threads(2)

import numpy as np
import pytest

import morph.render as renderer
from morph.recipe import Recipe


def reference_tone(x, get, scale, active):
    if not active:
        return x
    Lc = renderer._luma(x).clamp(0, 1.2).clamp(0, 1)
    edge = (4 * Lc * (1 - Lc)).clamp(min=1e-6)
    dh = get("dehaze")
    d = (get("contrast") + 0.35 * dh) * 0.55 * (Lc - 0.45) * edge
    d = d + get("highlights") * 0.30 * renderer._bump(Lc, 0.78, 0.15) * edge.sqrt()
    d = d + get("shadows") * 0.30 * renderer._bump(Lc, 0.22, 0.15) * edge.sqrt()
    d = d + get("whites") * 0.20 * renderer._bump(Lc, 0.94, 0.08)
    d = d + (get("blacks") - 0.25 * dh) * 0.15 * renderer._bump(Lc, 0.04, 0.07)
    for key, center, width in (("curve_shadows", 0.12, 0.08), ("curve_darks", 0.36, 0.11),
                               ("curve_lights", 0.62, 0.11), ("curve_highlights", 0.86, 0.08)):
        d = d + get(key) * 0.15 * renderer._bump(Lc, center, width) * edge.sqrt()
    return x + d


TONE_KEYS = ("contrast", "highlights", "shadows", "whites", "blacks", "curve_shadows",
             "curve_darks", "curve_lights", "curve_highlights")


@pytest.mark.parametrize("keys", [(), *[(k,) for k in TONE_KEYS], TONE_KEYS])
@pytest.mark.parametrize("local", [False, True])
def test_sparse_tone_matches_reference(monkeypatch, keys, local):
    img = np.random.default_rng(11).random((31, 43, 3), dtype=np.float32)
    img[0] = 0
    img[-1] = 1
    sliders = {k: (-0.3 if i % 2 else 0.4) for i, k in enumerate(keys)}
    recipe = Recipe(local={"face_skin": sliders}) if local else Recipe(sliders)
    masks = {"face_skin": np.full(img.shape[:2], 0.7, np.float32)} if local else None
    actual = renderer.render_recipe(img, masks, recipe)
    with monkeypatch.context() as m:
        m.setattr(renderer, "_adjust", reference_tone)
        expected = renderer.render_recipe(img, masks, recipe)
    np.testing.assert_array_equal(actual, expected)


def test_dense_tone_gradients_match_reference():
    x = torch.linspace(0, 1, 189).reshape(1, 3, 7, 9).requires_grad_()
    p = renderer.Params.from_recipe(Recipe({k: 0.2 for k in TONE_KEYS}), requires_grad=True)
    actual = renderer._adjust(x, p.G, 9, set(TONE_KEYS))
    expected = reference_tone(x, p.G, 9, set(TONE_KEYS))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ga = torch.autograd.grad(actual.sum(), (x, p.gv), retain_graph=True)
    ge = torch.autograd.grad(expected.sum(), (x, p.gv))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_inactive_parameters_preserve_values_and_gradients(requires_grad):
    x = torch.rand((1, 3, 7, 9), generator=torch.Generator().manual_seed(4))
    p = renderer.Params.from_recipe(Recipe({"shadows": 0.2, "curve_lights": -0.3}), requires_grad=requires_grad)
    actual = renderer._adjust(x, p.G, 9, {"contrast"})
    expected = reference_tone(x, p.G, 9, {"contrast"})
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if requires_grad:
        ga = torch.autograd.grad(actual.sum(), p.gv)[0]
        ge = torch.autograd.grad(expected.sum(), p.gv)[0]
        torch.testing.assert_close(ga, ge, atol=0, rtol=0)
