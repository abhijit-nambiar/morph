# Morph: scene edit reconstruction starter

This is a local command-line research prototype. It proposes photo groups, generates deterministic color-and-tone targets, fits a reproducible transformation, and transfers it to each group. It does not need an API key or alter image geometry. It does not yet understand free-form text, automatically normalize scenes, or establish professional quality. Review the outputs and proposed groups.

## Setup

Requires Python 3, NumPy and Pillow. In a terminal from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy Pillow
python test_pipeline.py
```

The tests use temporary synthetic images; they do not need the original dataset. This package does not contain your photos or derived images. Files and paths below are examples: replace `/path/to/originals` with your own album folder.

## Run an album

```bash
python morph.py group /path/to/originals --method chronological --gap-minutes 5 --out groups.json
python review_groups.py groups.json --source /path/to/originals --out groups-review
```

Inspect the contact sheets and edit `groups.json` if needed. A red border marks each group's representative. Groups use capture time/camera model and do not reliably recognize scenes. Images without capture times become single-photo groups. `groups-review/index.json` lists unassigned images.

```bash
python morph.py album groups.json \
  --target-engine pillow-soft --target-settings example-target-settings.json \
  --renderer gamut --solver gauss-newton --full-res-reference --out album-output
python review_album.py album-output/review.json --out album-review
```

This creates a deterministic target from each representative using independent per-channel tone curves and color-range adjustments, fits the bounded renderer, and applies its recovered settings to each original in the group. The single target settings file defines a shared test look; edit it to experiment with different combinations. Target controls and renderer controls have different meanings.

Each scene has `fit.json` with its target settings, recovered and applied recipe, and fit metrics. `target.jpg` and `reconstruction.jpg` are small previews used for fitting. `target-full.jpg` and `reconstruction-full.jpg` are full-resolution representative comparisons when `--full-res-reference` is used. Each `exports/` folder contains full-resolution edited photos. Source EXIF is preserved where readable, orientation is normalized, and output uses an sRGB ICC profile; other metadata such as XMP/IPTC may not be preserved.

`review.json` flags changes in near-black or near-white pixels before JPEG encoding. A flag is a prompt to inspect a photo, not proof of bad quality; no flag is not proof of good quality. The contact sheets are thumbnails: open the individual JPEGs for full-resolution review.

## Replay or refine a look

```bash
python morph.py apply groups.json album-output --out album-replay
```

This uses the stored fitted settings and renders from originals without fitting again. Add `--strength 0.7` to scale the recovered look toward neutral. To set intentional corrections, create a JSON file like:

```json
{
  "album": {"temperature": 0.02},
  "scenes": {"scene-01": {"shadows": 0.06}},
  "photos": {"scene-01": {"photo.jpg": {"exposure": 0.08}}}
}
```

Replace scene IDs and filenames with ones in your group file, then add `--corrections corrections.json` to `apply`. Operations are composed in order: per-photo correction, fitted scene look, scene correction, album correction. Corrections are manual and use the saved scene renderer's control definitions; polynomial mappings use the gamut renderer for the corrections. Each output scene records its applied recipe and corrections in `applied.json`.

## Other target and mapping options

- Fit one existing same-sized, aligned representative pair with `python morph.py fit ORIGINAL.jpg TARGET.jpg --renderer gamut --solver gauss-newton --out fit-output`.
- Use existing same-name, same-size representative targets across the groups with `python morph.py album groups.json --targets /path/to/edits --renderer gamut --solver gauss-newton --out album-from-targets`. Same dimensions do not guarantee alignment; inspect the pairs.
- `--renderer polynomial --solver least-squares --regularization .001` fits a deterministic nonlinear RGB color mapping rather than slider controls. It is global and can clip or overfit; compare its results with the bounded renderer.
- `--target-engine pillow-selective` creates a stronger target with independent global and hue-range operations. It can introduce clipping. The current `--full-res-reference` option is supported for `pillow-soft`, `morph`, and supplied targets, not the other Pillow target engines.

`group`, `target`, `fit`, `album`, `apply`, and `evaluate` refuse to overwrite their inputs or existing planned outputs. Use a new output directory, or pass `--overwrite` to replace planned outputs (this never allows overwriting an input and does not delete stale files). Full-resolution processing is sequential and can take minutes for a large album.
