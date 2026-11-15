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


def test_corrupt_file_is_skipped_not_fatal(fake_models, tiny_album, tmp_path):
    from morph.album import Album

    (tiny_album / "broken.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    a = Album(tiny_album, tmp_path / "cache").analyze()
    assert len(a.photos) == 7 and [n for n, _ in a.skipped] == ["broken.jpg"]


def test_project_upgrade_rebuilds_analysis_and_preserves_user_edits(fake_models, tiny_album, tmp_path, monkeypatch):
    from morph.album import ANALYSIS_VERSION, Album

    root = tmp_path / "project"
    project = Project.create(root, tiny_album)
    name = project.album.names[0]
    ref = project.adjust(name, {"exposure": 0.25, "face_skin.texture": -0.3})
    project.save_reference(ref, project.album.work_image(name))
    project.scenes[0].label = "My scene label"
    project.save()
    metadata = (root / "project.json").read_bytes()
    references = {p.name: p.read_bytes() for p in project.ref_dir.iterdir()}

    index = project.album.cache / "analysis.json"
    old = json.loads(index.read_text())
    old["version"] = "previous-analysis-version"
    index.write_text(json.dumps(old))
    analyze_one = Album._analyze_one
    rebuilt = []

    def track(self, info, *args):
        rebuilt.append(info.name)
        return analyze_one(self, info, *args)

    monkeypatch.setattr(Album, "_analyze_one", track)
    reopened = Project(root)
    assert sorted(rebuilt) == sorted(reopened.album.names) == sorted(project.album.names)
    assert json.loads(index.read_text())["version"] == ANALYSIS_VERSION
    assert (root / "project.json").read_bytes() == metadata
    assert {p.name: p.read_bytes() for p in reopened.ref_dir.iterdir()} == references
    assert reopened.reference(name).final.g("exposure") == 0.25
    assert reopened.scenes[0].label == "My scene label"

    rebuilt.clear()
    assert Project(root).album.names == reopened.album.names
    assert rebuilt == []  # opening a current cache stays cheap


def test_project_upgrade_reconciles_added_and_removed_photos(fake_models, tiny_album, tmp_path):
    from shutil import copyfile

    from morph.export import review_page
    from morph.scenes import Scene

    root = tmp_path / "project"
    project = Project.create(root, tiny_album)
    names = project.album.names
    project.scenes = [Scene("custom-first", names[:3], names[0], "Keep first"),
                      Scene("scene-01", names[3:4], names[3], "Removed scene"),
                      Scene("custom-last", names[4:], names[4], "Keep last")]
    project.save()
    project.adjust(names[1], {"exposure": 0.3})
    saved_ref = (project.ref_dir / f"{names[1][:-4]}.json").read_bytes()
    copyfile(tiny_album / names[1], tiny_album / "new-a.jpg")
    copyfile(tiny_album / names[4], tiny_album / "new-b.jpg")
    (tiny_album / names[0]).unlink()  # representative of a retained scene
    (tiny_album / names[3]).unlink()  # sole member of a removed scene
    index = project.album.cache / "analysis.json"
    data = json.loads(index.read_text())
    data["version"] = "previous-analysis-version"
    index.write_text(json.dumps(data))

    reopened = Project(root)
    all_members = [n for s in reopened.scenes for n in s.photos]
    assert len(all_members) == len(set(all_members)) == len(reopened.album.names)
    assert set(all_members) == set(reopened.album.names)
    first, last = reopened.scenes[:2]
    assert (first.id, first.label, first.photos, first.representative) == ("custom-first", "Keep first", names[1:3], names[1])
    assert (last.id, last.label, last.photos) == ("custom-last", "Keep last", names[4:])
    assert "scene-01" not in {s.id for s in reopened.scenes}
    assert len({s.id for s in reopened.scenes}) == len(reopened.scenes)
    assert (reopened.ref_dir / f"{names[1][:-4]}.json").read_bytes() == saved_ref
    html = review_page(reopened.album, reopened.scenes, {}, [], root).read_text()
    assert all(f"<b>{n}</b>" in html for n in reopened.album.names)
    assert "<b>" + names[0] + "</b>" not in html
    assert [s.to_json() for s in Project(root).scenes] == [s.to_json() for s in reopened.scenes]
