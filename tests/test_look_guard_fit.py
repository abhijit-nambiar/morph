import numpy as np

from morph.fit import fit_recipe
from morph.guard import check, warp_test_candidate
from morph.look import parse
from morph.recipe import Recipe
from morph.render import render_recipe
from tests.conftest import FakeSegmenter, make_photo
from morph.imageio import load_rgb


def test_parse_regions_and_modifiers():
    r = parse("warm golden hour, soft skin, slightly darker background, brighter eyes").recipe
    assert r.g("temperature") > 0.2
    assert r.l("background", "exposure") < 0 and r.l("background", "exposure") > -0.35
    assert r.l("face_skin", "clarity") < 0 and r.l("body_skin", "clarity") < 0
    assert r.l("eyes", "exposure") > 0


def test_parse_negation_and_unknown():
    p = parse("less contrast, no noise, flibbertigibbet")
    assert p.recipe.g("contrast") < 0 and p.recipe.g("noise_reduction") > 0
    assert "flibbertigibbet" in p.unknown


def _photo(tmp_path):
    return load_rgb(make_photo(tmp_path / "p.jpg", 3, size=(320, 220)))


def test_guard_accepts_colour_edit_rejects_structure(tmp_path):
    img = _photo(tmp_path)
    m = FakeSegmenter().segment((img * 255).astype(np.uint8)).masks
    good = render_recipe(img, m, Recipe({"exposure": 0.4, "temperature": 0.3}, {"background": {"saturation": -0.4}}))
    assert check(img, good)[0].verdict == "pass"
    assert check(img, warp_test_candidate(img, "replace"))[0].verdict == "reject"
    assert check(img, warp_test_candidate(img, "liquify"))[0].verdict != "pass"
    rep, aligned, _ = check(img, warp_test_candidate(good, "shift"))
    assert rep.alignment in ("homography", "ecc") and rep.verdict != "reject"


def test_fit_recovers_look(tmp_path):
    img = _photo(tmp_path)
    m = FakeSegmenter().segment((img * 255).astype(np.uint8)).masks
    truth = Recipe({"exposure": 0.4, "temperature": 0.3, "contrast": 0.2}, {"background": {"exposure": -0.4}, "face_skin": {"tint": 0.2}})
    target = render_recipe(img, m, truth)
    res = fit_recipe(img, target, m, steps=160, max_side=200)
    assert res.de_before > 4 and res.de_after < 1.5
    assert res.recipe.g("temperature") > 0.1


def test_fit_keeps_global_match_when_local_optimization_overshoots(tmp_path):
    img = load_rgb(make_photo(tmp_path / "p.jpg", 0, size=(320, 220)))
    masks = FakeSegmenter().segment((img * 255).astype(np.uint8)).masks
    target = render_recipe(img, masks, Recipe({
        "exposure": 0.25, "temperature": 0.12, "contrast": 0.08,
        "saturation": 0.06, "highlights": -0.12, "shadows": 0.08,
    }))

    result = fit_recipe(img, target, masks, steps=80, max_side=200)

    assert result.de_after < 0.35
    assert not result.recipe.local
