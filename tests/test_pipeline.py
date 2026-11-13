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
