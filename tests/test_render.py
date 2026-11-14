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


def test_clip_budget_holds_target_to_reference_clipping():
    import torch
    from morph.propagate import PropagationConfig, _correct, _overshoot, _prep, clip_added
    from morph.recipe import Recipe
    from morph.render import Params, render
    from morph.colorspace import srgb_to_lab

    rng = np.random.default_rng(0)
    img = np.clip(0.55 + 0.25 * rng.random((96, 128, 3)), 0, 1).astype(np.float32)
    hot = Recipe({"exposure": 1.2})
    assert clip_added(img, {}, Recipe({"contrast": 0.1})) < 0.01 < clip_added(img, {}, hot)

    x, mt = _prep(img, {}, 128)
    with torch.no_grad():
        base_lab = srgb_to_lab(render(x, mt, Params.from_recipe(hot)).clamp(0, 1).permute(0, 2, 3, 1))[0].mean((0, 1))
    cfg = PropagationConfig(steps=40, refine_steps=2, side=128)

    def clipped(rec):
        with torch.no_grad():
            return float(_overshoot(render(x, mt, Params.from_recipe(rec), gamut_map=False)))

    free = _correct(img, {}, hot, {"global": base_lab.tolist()}, cfg)
    held = _correct(img, {}, hot, {"global": base_lab.tolist()}, cfg, clip_budget=0.0)
    assert clipped(held) < clipped(free) - 0.05


def test_desaturating_reference_without_face_keeps_target_faces_neutral():
    from types import SimpleNamespace
    from morph.propagate import PropagationConfig, Reference, _targets
    from morph.recipe import Recipe

    def st(**regions):
        return {r: {"mean": m, "std": s, "pct": [0, 50, 100], "coverage": 0.2} for r, (m, s) in regions.items()}

    ref = Reference("ref", Recipe({"saturation": -1.0}), "s0",
                    st(**{"global": ([57, -16, 27], [20, 12, 15])}),
                    st(**{"global": ([50, 0, 0], [22, 0.1, 0.1])}))
    photo = SimpleNamespace(name="p", stats=st(**{"global": ([55, 5, 10], [20, 10, 10]), "face_skin": ([60, 14, 22], [8, 3, 3])}))
    album = SimpleNamespace(photos=[photo])
    t = _targets(album, [ref], np.ones((1, 1)), [SimpleNamespace(id="s0", photos=["ref"]), SimpleNamespace(id="s1", photos=["p"])],
                 PropagationConfig())[0]
    assert np.hypot(*t["face_skin"][1:]) < 1.0 and np.hypot(*t["global"][1:]) < 1.0


def test_grey_reference_does_not_desaturate_targets():
    from morph.propagate import Reference, _chroma_gain
    from morph.recipe import Recipe

    grey = {"global": {"mean": [50, 0, 0], "std": [20, 0.05, 0.05], "pct": [0, 50, 100], "coverage": 1.0}}
    assert np.allclose(_chroma_gain(Reference("g", Recipe(), "", grey, grey), "global"), 1.0)


def test_flat_coloured_reference_turned_bw_still_desaturates():
    from morph.propagate import Reference, _chroma_gain
    from morph.recipe import Recipe

    st = lambda m, sd: {"global": {"mean": m, "std": sd, "pct": [0, 50, 100], "coverage": 1.0}}
    ref = Reference("f", Recipe({"saturation": -1}), "", st([50, -16, -20], [20, 0.5, 0.5]), st([50, 0, 0], [20, 0.01, 0.01]))
    assert np.all(_chroma_gain(ref, "global")[1:] < 0.05)
||||||| parent of 1011cf3 (Segmentation: gate people parts by the scene model's person map (drops costume-like objects labelled clothes))




def test_people_gate_drops_parts_the_scene_model_does_not_call_people():
    from morph.segment import _gate_people

    z = np.zeros((100, 100), np.float32)
    clothes = z.copy(); clothes[10:40, 10:40] = 1; clothes[60:90, 60:90] = 1  # a person and a carousel horse
    human = {"background": 1 - clothes, "hair": z, "face": z, "glasses": z, "body_skin": z, "clothes": clothes}
    scene_person = z.copy(); scene_person[10:40, 10:40] = 0.9
    g = _gate_people(human, scene_person)
    assert g["clothes"][25, 25] > 0.9 and g["clothes"][75, 75] < 0.05 and g["background"][75, 75] > 0.95
    assert _gate_people(human, z)["clothes"][75, 75] == 1  # scene model sees nobody: leave the parser alone
