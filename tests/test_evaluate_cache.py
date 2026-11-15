import hashlib
import shutil

import numpy as np

from morph.album import Album
from morph import evaluate as evaluation
from morph.imageio import save_png_u8


def test_aligned_targets_invalidate_legacy_colour_cache(tiny_album, tmp_path, fake_models, monkeypatch):
    album = Album(tiny_album).analyze()
    edits = tmp_path / 'edits'
    edits.mkdir()
    edit = edits / 'a0.jpg'
    shutil.copy2(tiny_album / edit.name, edit)
    cache = tmp_path / 'aligned'
    cache.mkdir()
    st = edit.stat()
    legacy = hashlib.sha1(f'{edit.resolve()}:{st.st_size}:{st.st_mtime_ns}'.encode()).hexdigest()[:10]
    shape = album.work_image(edit.name).shape
    save_png_u8(np.zeros(shape, np.float32), cache / f'a0.{legacy}.png')
    np.save(cache / f'a0.{legacy}.w.npy', np.ones(shape[:2], np.float16))

    targets = evaluation._aligned_edits(album, edits, cache)
    assert targets[edit.name][0].mean() > .05
    calls = []
    original = evaluation.align

    def align(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, 'align', align)
    evaluation._aligned_edits(album, edits, cache)
    assert calls == []
    monkeypatch.setattr(evaluation, 'ANALYSIS_VERSION', 'future-colour-renderer')
    evaluation._aligned_edits(album, edits, cache)
    assert len(calls) == 1
