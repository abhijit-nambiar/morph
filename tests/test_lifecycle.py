import json
import os
import shutil
import stat

import numpy as np
import pytest

from morph import album as album_mod

from morph.album import Album
from morph.cli import main
from morph.project import Project
from morph.studio import State
from tests.conftest import make_photo


@pytest.fixture
def small_album(tmp_path, fake_models):
    folder = tmp_path / "album"
    folder.mkdir()
    for i in range(3):
        make_photo(folder / f"a{i}.jpg", i, size=(96, 64))
    return folder


def assert_scenes_cover_album(project):
    members = [name for scene in project.scenes for name in scene.photos]
    assert len(members) == len(set(members))
    assert set(members) == set(project.album.names)
    assert all(scene.representative in scene.photos for scene in project.scenes)


@pytest.mark.parametrize("delete_reference", [False, True])
def test_delete_reconciles_and_exports(small_album, tmp_path, delete_reference):
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    project.adjust("a0.jpg", {"exposure": 0.1})
    project.adjust("a1.jpg", {"exposure": 0.2})
    saved_refs = {p: p.read_bytes() for p in project.ref_dir.glob("*.json")}
    removed = "a0.jpg" if delete_reference else "a2.jpg"
    (small_album / removed).unlink()

    reopened = Project(root)
    assert set(reopened.album.names) == {"a0.jpg", "a1.jpg", "a2.jpg"} - {removed}
    assert_scenes_cover_album(reopened)
    assert {r.name for r in reopened.references()} == {"a0.jpg", "a1.jpg"} - {removed}
    assert all(path.read_bytes() == data for path, data in saved_refs.items())
    out = tmp_path / "out"
    main(["apply", str(root), "--out", str(out)])
    assert {p.name for p in (out / "photos").glob("*.jpg")} == set(reopened.album.names)


def test_add_preserves_scenes_and_only_analyzes_new(small_album, tmp_path, monkeypatch):
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    for i, scene in enumerate(project.scenes):
        scene.label = f"Label {i}"
    project.save()
    original_scenes = [s.to_json() for s in project.scenes]
    make_photo(small_album / "new.jpg", 50, size=(96, 64))
    analyzed = []
    original = Album._analyze_one

    def track(self, info, *args):
        analyzed.append(info.name)
        return original(self, info, *args)

    monkeypatch.setattr(Album, "_analyze_one", track)
    reopened = Project(root)
    assert analyzed == ["new.jpg"]
    assert set(reopened.album.names) == set(project.album.names) | {"new.jpg"}
    assert [s.to_json() for s in reopened.scenes[:len(original_scenes)]] == original_scenes
    assert_scenes_cover_album(reopened)
    assert Project(root).album.names == reopened.album.names
    assert analyzed == ["new.jpg"]


@pytest.mark.parametrize("completed", [0, 2])
def test_interrupted_analysis_reuses_completed(small_album, tmp_path, monkeypatch, completed):
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    for i in range(3):
        make_photo(small_album / f"new{i}.jpg", 50 + i, size=(96, 64))
    original = Album._analyze_one
    analyzed = []

    def interrupt(self, info, *args):
        if len(analyzed) == completed:
            raise KeyboardInterrupt
        result = original(self, info, *args)
        analyzed.append(info.name)
        return result

    monkeypatch.setattr(Album, "_analyze_one", interrupt)
    with pytest.raises(KeyboardInterrupt):
        Project(root)
    assert len(analyzed) == completed
    partial = Album.load(small_album, project.album.cache)
    assert len(partial.photos) == 3 + completed
    analyzed.clear()

    def track(self, info, *args):
        analyzed.append(info.name)
        return original(self, info, *args)

    monkeypatch.setattr(Album, "_analyze_one", track)
    reopened = Project(root)
    assert analyzed == [f"new{i}.jpg" for i in range(completed, 3)]
    assert len(reopened.album.photos) == 6
    assert_scenes_cover_album(reopened)


def test_studio_reopen_refreshes_photos(small_album, tmp_path):
    root = tmp_path / "project"
    Project.create(root, small_album)
    (small_album / "a0.jpg").unlink()
    make_photo(small_album / "new.jpg", 50, size=(96, 64))
    assert set(State(root).state()["photos"]) == {"a1.jpg", "a2.jpg", "new.jpg"}


def test_corrupt_file_does_not_repeat_analysis_on_reopen(small_album, tmp_path, monkeypatch):
    (small_album / "broken.jpg").write_bytes(b"not a jpeg")
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    assert [name for name, _ in project.album.skipped] == ["broken.jpg"]
    calls = []

    def unexpected(self, *args, **kwargs):
        calls.append(self)
        raise AssertionError("reopening an unchanged album must not analyze")

    monkeypatch.setattr(Album, "analyze", unexpected)
    monkeypatch.setattr(Album, "_analyze_one", unexpected)
    for _ in range(2):
        reopened = Project(root)
        assert reopened.album.names == project.album.names
        assert reopened.album.skipped == project.album.skipped
    assert calls == []


def test_same_name_overwrite_refreshes_analysis(small_album, tmp_path):
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    old_sha = project.album.get("a2.jpg").sha256
    old_work = project.album.work_image("a2.jpg").copy()
    make_photo(small_album / "a2.jpg", 99, gain=0.5, size=(96, 64))
    reopened = Project(root)
    assert reopened.album.get("a2.jpg").sha256 != old_sha
    assert not np.array_equal(reopened.album.work_image("a2.jpg"), old_work)


def test_repaired_skipped_file_joins_album(small_album, tmp_path):
    root = tmp_path / "project"
    broken = small_album / "broken.jpg"
    broken.write_bytes(b"truncated jpeg")
    Project.create(root, small_album)
    make_photo(broken, 99, size=(96, 64))
    reopened = Project(root)
    assert "broken.jpg" in reopened.album.names
    assert reopened.album.skipped == []
    assert_scenes_cover_album(reopened)


def test_interrupted_analysis_retains_duplicate_names(small_album, tmp_path, monkeypatch):
    shutil.copy2(small_album / 'a0.jpg', small_album / 'b-copy.jpg')
    project = Project.create(tmp_path / 'project', small_album)

    def interrupt(i, total, name):
        if name == 'a1.jpg':
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Album(small_album, project.album.cache).analyze(progress=interrupt)
    assert Album.load(small_album, project.album.cache).names == project.album.names
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(args)
        raise AssertionError('cached duplicate must not be read')

    monkeypatch.setattr(album_mod, 'read_info', unexpected)
    resumed = Album(small_album, project.album.cache).analyze()
    assert resumed.names == project.album.names
    assert calls == []


def test_interrupted_early_addition_retains_unvisited_cache(small_album, tmp_path, monkeypatch):
    for path in small_album.glob("a*.jpg"):
        path.rename(path.with_name("b" + path.name[1:]))
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    for i in range(2):
        make_photo(small_album / f"a{i}.jpg", 50 + i, size=(96, 64))
    original = Album._analyze_one

    def interrupt(self, info, *args):
        if info.name == "a1.jpg":
            raise KeyboardInterrupt
        return original(self, info, *args)

    monkeypatch.setattr(Album, "_analyze_one", interrupt)
    with pytest.raises(KeyboardInterrupt):
        Project(root)
    analyzed = []

    def track(self, info, *args):
        analyzed.append(info.name)
        return original(self, info, *args)

    monkeypatch.setattr(Album, "_analyze_one", track)
    reads = []
    read_info = album_mod.read_info

    def track_read(path):
        reads.append(path.name)
        return read_info(path)

    monkeypatch.setattr(album_mod, "read_info", track_read)
    reopened = Project(root)
    assert analyzed == ["a1.jpg"]
    assert reads == ["a1.jpg"]
    assert set(reopened.album.names) == set(project.album.names) | {"a0.jpg", "a1.jpg"}


def test_unchanged_reopen_does_not_read_info(small_album, tmp_path, monkeypatch):
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unchanged sources must not be read")

    monkeypatch.setattr(album_mod, "read_info", unexpected)
    monkeypatch.setattr(Album, "_analyze_one", unexpected)
    assert Project(root).album.names == project.album.names
    assert Album(small_album, project.album.cache).analyze().names == project.album.names
    assert calls == []


def test_old_index_without_stat_loads(small_album, tmp_path):
    root = tmp_path / "project"
    project = Project.create(root, small_album)
    index = project.album.cache / "analysis.json"
    data = json.loads(index.read_text())
    data.pop("stat", None)
    index.write_text(json.dumps(data))
    assert Album.load(small_album, project.album.cache).names == project.album.names
    assert Project(root).album.names == project.album.names


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permission metadata')
def test_restored_read_permission_retries_skipped_photo(small_album, tmp_path, monkeypatch):
    blocked = small_album / 'a2.jpg'
    mode = blocked.stat().st_mode
    blocked.chmod(mode & ~stat.S_IRUSR)
    denied = True
    original = album_mod.read_info

    def read(path):
        if path == blocked and denied:
            raise PermissionError('read access denied')
        return original(path)

    monkeypatch.setattr(album_mod, 'read_info', read)
    root = tmp_path / 'project'
    project = Project.create(root, small_album)
    assert 'a2.jpg' not in project.album.names
    before = blocked.stat()
    blocked.chmod(mode)
    denied = False
    after = blocked.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    reopened = Project(root)
    assert 'a2.jpg' in reopened.album.names
    assert reopened.album.skipped == []
