import numpy as np
import pytest
import torch

import morph.render as renderer
from morph.recipe import GLOBAL_KEYS, LOCAL_KEYS, LOCAL_REGIONS, Recipe


@pytest.mark.parametrize('keys', [None, ['exposure'], ['texture'], ['saturation']])
def test_local_strips_match_untiled_pixels_and_gradients(monkeypatch, keys):
    def untiled(x, get, scale, active):
        x = renderer._adjust_tone(x, get, scale, active)
        x = renderer._adjust_detail(x, get, scale, active)
        return renderer._adjust_color(x, get, scale, active)

    rng = np.random.default_rng(31)
    img = rng.uniform(0.15, 0.85, (48, 64, 3)).astype(np.float32)
    recipe = Recipe(
        {k: 0.05 for k in (keys or GLOBAL_KEYS) if k != "grain"},
        {r: {k: 0.04 for k in (keys or LOCAL_KEYS)} for r in ("background", "face_skin", "clothes")},
    )
    masks = {r: rng.uniform(0, 1, img.shape[:2]).astype(np.float32) for r in recipe.local}
    mt = renderer.masks_tensor(masks, LOCAL_REGIONS)
    x = renderer.to_tensor(img)
    weights = torch.linspace(0.5, 1.5, x.numel()).reshape_as(x)

    def run():
        p = renderer.Params.from_recipe(recipe, requires_grad=True)
        out = renderer.render(x, mt, p, active_global=set(recipe.global_),
                              active_local={LOCAL_REGIONS.index(r): set(v) for r, v in recipe.local.items()})
        grads = torch.autograd.grad((out * weights).mean(), (p.gv, p.lv))
        return out, grads

    monkeypatch.setattr(renderer, "_POINTWISE_PIXELS", 512)
    actual, actual_grads = run()
    monkeypatch.setattr(renderer, "_adjust", untiled)
    expected, expected_grads = run()
    torch.testing.assert_close(actual, expected, atol=1 / 65535, rtol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
