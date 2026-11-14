import json

import numpy as np
from PIL import Image

from morph.cli import main
from morph.project import Project
from morph.propagate import make_reference, propagate
from morph.export import export_album


def test_end_to_end(fake_models, tiny_album, tmp_path):
    proj = Project.create(tmp_path / "proj", tiny_album)
    assert len(proj.album.photos) == 7
    assert 1 <= len(proj.scenes) <= 4
    sugg = proj.suggest(2)
    assert len(sugg) == 2
    cands = proj.generate(sugg[0].name, "warm, soft skin, darker background", "local", 3)
    assert len(cands) == 3 and all(rep.verdict != "reject" for _, rep in cands)
    ref = proj.pick(sugg[0].name, 0)
    assert ref.recipe.g("temperature") > 0
    proj.adjust(sugg[0].name, {"face_skin.texture": -0.3, "exposure": 0.1})
    assert proj.reference(sugg[0].name).final.l("face_skin", "texture") < 0

    refs = [make_reference(proj.album, r.name, r.final, proj.scene_of(r.name)) for r in proj.references()]
    recipes = propagate(proj.album, refs, proj.scenes)
    assert set(recipes) == set(proj.album.names)
    report = export_album(proj.album, recipes, tmp_path / "out")
    for name, v in report.items():
        assert v["ok"] and v["same_geometry"]
        src, dst = Image.open(tiny_album / name), Image.open(tmp_path / "out" / "photos" / name)
        assert src.size == dst.size
        assert dst.getexif().get(0x010F) == "TestCam"


def test_propagation_homogenizes(fake_models, tiny_album, tmp_path):
    proj = Project.create(tmp_path / "proj", tiny_album)
    a_photos = [p.name for p in proj.album.photos if p.name.startswith("a")]
    ref_name = a_photos[0]
    proj.adjust(ref_name, {"exposure": 0.3, "temperature": 0.2})
    refs = [make_reference(proj.album, r.name, r.final, proj.scene_of(r.name)) for r in proj.references()]
    recipes = propagate(proj.album, refs, proj.scenes)
    from morph.render import render_recipe

    def key(n, rec=None):
        img = proj.album.work_image(n)
        out = img if rec is None else render_recipe(img, proj.album.masks(n).masks, rec)
        return out.mean()

    before = np.std([key(n) for n in a_photos])
    after = np.std([key(n, recipes[n]) for n in a_photos])
    assert after < before  # exposure differences inside the scene shrink


def test_cli_flow(fake_models, tiny_album, tmp_path, capsys):
    p = tmp_path / "proj"
    main(["init", str(p), str(tiny_album)])
    main(["suggest", str(p), "-k", "2"])
    first = capsys.readouterr().out.split("\n")
    name = next(l for l in first if l.startswith("1.")).split()[1]
    main(["generate", str(p), name, "--prompt", "airy, lifted shadows", "-n", "2"])
    main(["pick", str(p), name, "0"])
    main(["adjust", str(p), name, "hair.saturation=0.2"])
    main(["apply", str(p), "--out", str(tmp_path / "out")])
    v = json.loads((tmp_path / "out" / "verification.json").read_text())
    assert len(v) == 7 and all(x["ok"] for x in v.values())
    assert (tmp_path / "out" / "index.html").exists()


def test_references_skip_photos_no_longer_in_album(fake_models, tiny_album, tmp_path):
    proj = Project.create(tmp_path / "proj", tiny_album)
    name = proj.album.photos[0].name
    proj.adjust(name, {"exposure": 0.2})
    (tiny_album / name).unlink()  # e.g. a re-cull dropped it
    proj = Project.create(tmp_path / "proj", tiny_album)
    assert name not in proj.album.names and proj.references() == []


def test_export_verify_tolerates_tone_but_not_geometry():
    import cv2
    from morph.export import verify
    from morph.recipe import Recipe
    from morph.render import render_recipe

    rng = np.random.default_rng(0)
    img = np.zeros((480, 720, 3), np.float32) + np.linspace(0.15, 0.6, 720, dtype=np.float32)[None, :, None]
    for _ in range(40):
        x, y, r = rng.integers(20, 700), rng.integers(20, 460), int(rng.integers(5, 40))
        cv2.circle(img, (int(x), int(y)), r, rng.uniform(0.05, 0.95, 3).tolist(), -1)
    graded = render_recipe(img, None, Recipe({"exposure": -0.5, "highlights": -0.6, "shadows": 0.4, "contrast": 0.3,
                                              "temperature": 0.3, "dehaze": 0.2, "noise_reduction": 0.4}))
    assert verify(img, graded)["ok"]
    extreme = render_recipe(img, None, Recipe({"exposure": 1.0, "contrast": -0.8, "temperature": -1.0,
                                               "noise_reduction": 1.0, "texture": -1.0, "clarity": -1.0, "dehaze": 1.0}))
    assert verify(img, extreme)["ok"], verify(img, extreme)
    flat = (0.45 + 0.04 * (img - 0.5)).astype(np.float32)
    assert verify(flat, render_recipe(flat, None, Recipe({"exposure": -3.0})))["ok"]
    rot = cv2.warpAffine(graded, cv2.getRotationMatrix2D((360, 240), 1.5, 1.0), (720, 480), borderMode=cv2.BORDER_REFLECT)
    assert not verify(img, rot)["ok"]
    assert not verify(img, graded[:, ::-1].copy())["ok"]


def test_add_grain_matches_full_render():
    from morph.recipe import Recipe
    from morph.render import add_grain, render_recipe

    img = np.random.default_rng(1).uniform(0.1, 0.9, (64, 96, 3)).astype(np.float32)
    full = render_recipe(img, None, Recipe({"exposure": 0.3, "grain": 0.6}), seed=7)
    split = add_grain(render_recipe(img, None, Recipe({"exposure": 0.3}), seed=7), 0.6, seed=7)
    assert np.abs(full - split).max() < 1e-5
