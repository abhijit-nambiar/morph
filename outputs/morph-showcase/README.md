# Morph: styled album and normalization showcase

Open `index.html` in a browser. The showcase works offline; keep the whole output folder together, including `assets/`, `models/`, `varied-album/`, `final-albums/`, `local-experiment/` and the linked reports. It includes an eight-step pipeline walkthrough with scene/photo/style selection, actual fitting and correction evidence, three looks, a before/after slider, scene transfer comparisons, normalization measurements, scene filters and an inspector with five processing stages. A separate local-light section explains face/person workflows and shows an original, soft person mask, overlay and person-aware export from the earlier 13-photo experiment. Those local operations are not applied to the three styled albums. The **Present** button enables section navigation with left/right arrows; Escape exits. Gallery arrows navigate photos; Escape closes the inspector.

The album has 88 photos in 14 agent-reviewed lighting groups. One representative per group defines the intended baseline and supplies a styled fitting target. The remaining 74 photos receive the saved mapping without style refitting. A singleton shaded close-up is separately calibrated and does not demonstrate scene transfer.

## Canonical outputs

- `RESULTS.md`: measurements, verification and limitations.
- `PRESENTER-NOTES.md`: the eight-step guided demonstration and explanation of the local-light branch.
- `local-experiment/`: one copied historical person-aware export, explicitly separate from album results.
- `walkthrough-artifact-verification.json` and `walkthrough-browser-qa.json`: checks for the expanded walkthrough.
- `final-albums/{sunlit,coastal,soft-film}/`: 264 styled varied-photo exports plus 42 fitting-reference exports. `run.json` records completion. `replay-groups.json` includes the references required by existing input integrity checks.
- `models/{style}/targets/` and `models/{style}/fits/`: independent full-resolution reference targets and saved scene mappings.
- `varied-album/`: full-resolution simulated inputs, designated references and evaluation-only injected settings (seed `20261030`).
- `normalization-final.json`: matched-image correction estimates and graph diagnostics; `corrections-final.json`: equivalent recipes consumed by the existing renderer.
- `clean-baseline-hybrid.json`: evaluation-only baseline from applying the same normalizer to the clean album.
- `evaluation.json`: final normalization and decoded styled JPEG measurements, including noncalibration photos separately.
- `verification.json`: source identity, export coverage, decoding, dimensions and metadata presence.
- `artifact-verification.json`: complete showcase image/link checks; `walkthrough-browser-qa.json` records 43 passing checks for the current version. `browser-qa.json` retains the original 20-check report. `qa/` contains screenshots and per-scene output overview sheets.
- `benchmark-final.json`: renderer comparison on 14 fitting, 12 validation and 62 test frames per style.
- `groups.json` and `styles.json`: reviewed assignments and independent target settings.

Other variants, logs and previews in this folder document development. Files with `final` in their names and the paths above are the current results. The earlier `albums/` directory is a failed preflight and is not the delivered album.

## Serve after cloning

From the repository root, run `git lfs pull`, then `python3 -m http.server 8765 --bind 127.0.0.1 --directory outputs/morph-showcase`. Open `http://127.0.0.1:8765/#process`. Serving the saved site needs only Python 3; image-processing dependencies are needed only for regeneration or verification. See [the repo README](../../README.md) for clone/setup instructions. Saved reports use repo-relative paths; the website uses links relative to this folder. `portability-verification.json` records a relocated-copy check of all 1,334 website files and 306 full-resolution exports.

## Reproduction

From the repository root, use a fresh output directory:

```bash
.venv/bin/python -m pip install -r outputs/morph-starter/requirements-normalization.txt
.venv/bin/python outputs/morph-starter/showcase_workflow.py Morph/originals \
  --scenes outputs/morph-starter/showcase-scenes.json \
  --styles outputs/morph-starter/showcase-styles.json \
  --seed 20261030 --jobs 3 --out outputs/morph-showcase-reproduction
```

Add `--person-experiment outputs/morph-resumed` to attach the same separate historical person example when reproducing this checkout. Without that optional input, the walkthrough still explains the local branch, but no historical mask/output images are copied. Use `--jobs 1` to reduce memory. Saved file references in this showcase are relative to the repository root; run processing commands there. The HTML and its relative assets/export links can be moved together. Older research folders outside this showcase may retain historical machine paths.

## Interpretation

The normalizer sees varied photos and one intended reference per group, without injected settings or other clean counterparts. It estimates bounded global tone/color corrections using verified overlapping content. The independent evaluator compares the result with an algorithm-derived clean baseline. This establishes restoration under controlled synthetic variation, not ideal artistic normalization. References establish each scene's absolute exposure and color; the shared style recipe supplies album intent. It does not infer absolute calibration between unrelated scenes.

The synthetic variance family fits the corrector's global model. Local shadows, clipping recovery, subject masks, human-edit matching and unseen-album quality require separate work. The 88 supplied human edits remain preserved and were not used for this showcase.
