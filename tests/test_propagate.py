from types import SimpleNamespace

import numpy as np
import pytest

from morph.propagate import PropagationConfig, Reference, _adaptation_confidence
from morph.recipe import Recipe


def _album(highlight=98, lightness=75):
    return SimpleNamespace(
        names=["photo", "near", "far"],
        photos=[SimpleNamespace(stats={"global": {"pct": [0, 50, highlight], "mean": [lightness, 0, 0]}})],
        emb_background=np.array([[1, 0, 0], [0.95, np.sqrt(1 - 0.95**2), 0], [0.35, 0, np.sqrt(1 - 0.35**2)]]),
    )


def _ref(name="far", recipe=None, **kwargs):
    return Reference(name, recipe or Recipe(), orig_stats={"global": {"mean": [45, 0, 0], "std": [15, 5, 5]}}, **kwargs)


def test_new_backdrop_keeps_look_without_forcing_reference_statistics():
    album = _album()
    cfg = PropagationConfig()
    near, far = _ref("near"), _ref("far")
    confidence = lambda ref: _adaptation_confidence(album, 0, [ref], np.ones(1), np.zeros(1), cfg)
    assert confidence(near) == 1
    assert confidence(far) == 0
    assert _adaptation_confidence(album, 0, [near, far], np.array([0.25, 0.75]), np.zeros(2), cfg) == pytest.approx(0.25)


def test_dark_photo_and_scene_reference_keep_exposure_adaptation():
    ref = _ref()
    cfg = PropagationConfig()
    assert _adaptation_confidence(_album(70), 0, [ref], np.ones(1), np.zeros(1), cfg) == 1
    assert _adaptation_confidence(_album(), 0, [ref], np.ones(1), np.ones(1), cfg) == 1


def test_exposure_confidence_changes_smoothly_across_burst_frames():
    ref = _ref()
    cfg = PropagationConfig()
    confidence = lambda highlight: _adaptation_confidence(_album(highlight), 0, [ref], np.ones(1), np.zeros(1), cfg)
    assert confidence(90) == 1
    assert confidence(92.5) == pytest.approx(0.5)
    assert confidence(95) == 0
    assert abs(confidence(90.01) - confidence(89.99)) < 0.003


def test_bright_highlights_alone_do_not_disable_backlit_portrait_correction():
    cfg = PropagationConfig()
    assert _adaptation_confidence(_album(100, 40), 0, [_ref()], np.ones(1), np.zeros(1), cfg) == 1
    # Near the edge of the reference's brightness distribution, fade protection smoothly.
    assert _adaptation_confidence(_album(100, 62.5), 0, [_ref()], np.ones(1), np.zeros(1), cfg) == pytest.approx(0.5)


def test_dissimilar_scene_still_receives_reference_look_and_clip_budget(monkeypatch):
    import morph.propagate as p

    album = _album()
    album.photos[0].name = "photo"
    album.work_image = lambda name: np.zeros((8, 8, 3), np.float32)
    album.masks = lambda name: SimpleNamespace(masks={})
    monkeypatch.setattr(p, "reference_weights", lambda *args: np.ones((1, 1)))
    monkeypatch.setattr(p, "_targets", lambda *args: [{"global": [50, 0, 0]}])
    seen = []

    def correct(img, masks, base, targets, cfg, gain, budget):
        seen.append((base.g("temperature"), gain, budget))
        return base

    monkeypatch.setattr(p, "_correct", correct)
    ref = _ref("far", Recipe({"temperature": 0.3}), clip_added=0.02)
    # An omitted Reference.scene is resolved from the supplied scene membership.
    scenes = [SimpleNamespace(id="a", photos=["photo"]), SimpleNamespace(id="b", photos=["far"])]
    result = p.propagate(album, [ref], scenes)
    assert seen == [(pytest.approx(0.3), 0, 0.02)]
    assert result["photo"].g("temperature") == pytest.approx(0.3)
    assert result["photo"].meta["adaptation_confidence"] == 0
    scenes[0].photos.append("far")
    scenes.pop()
    p.propagate(album, [ref], scenes, p.PropagationConfig(pursue_within=0.8))
    assert seen[-1][1] == pytest.approx(0.8)
