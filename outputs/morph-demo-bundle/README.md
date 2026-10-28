# Morph: deterministic scene edit reconstruction demo

This local prototype turns a supported text prompt into a deterministic target edit, reconstructs a reusable color-and-lighting mapping from one photo, and transfers the mapping to other photos in the same scene. It uses no API key and leaves image geometry unchanged.

## Show the result

For a quick presentation, open `showcase/01-target.png`, `showcase/02-reconstruction.png`, and `showcase/03-transfer.png` in order. These are 1920 × 1080 presentation slides; the individual JPEGs below remain full resolution.

1. Open `originals/DSC_0906-4.jpg`, then `results/representative-target.jpg`. The target was generated from the prompt **“warm, slightly soft contrast and lift shadows.”**
2. Open `results/representative-reconstruction.jpg`. The program fitted its own deterministic color-and-lighting mapping to the original and target.
3. Compare `originals/DSC_0901-1.jpg` with `results/DSC_0901-1.jpg`, and `originals/DSC_0928-10.jpg` with `results/DSC_0928-10.jpg`. These are different photos that received the same recovered scene mapping.
4. Open `results/recovered-edit.json` to show the saved fitted settings and the interpreted target settings. These are reproducible operations, not the original editor's ground-truth sliders.

All displayed photos are full-resolution JPEGs: 6048 × 4032 for the three scene photos. JPEG export is lossy. The saved fit was computed with images reduced to a maximum side of 320 pixels.

For this synthetic target, reduced-resolution pre-JPEG RMSE against the independently generated target fell from 0.02719 to 0.00198 on `DSC_0928-10.jpg`, and from 0.02898 to 0.00199 on `DSC_0901-1.jpg`. Those numbers measure pixel reconstruction of this particular target, not professional photographic quality.

## Repeat the demo

Requires Python 3, NumPy, and Pillow. From this folder:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install numpy Pillow
./run-demo.sh
```

`run-demo.sh` fits the representative, creates full-resolution target and reconstruction files, and renders all three scene samples into a new `demo-output/` folder. It will not overwrite an existing output. Give a different folder name to run again, for example `./run-demo.sh another-run`.

The file `groups.json` defines the representative and its scene. `code/` contains the local prototype. `demo-output/scene-01/fit.json` contains the fitted mapping; `demo-output/review.json` and `report.json` contain the transfer checks. `python3 code/morph.py prompt-help` lists the fixed supported vocabulary.

## Honest scope

The bundle showcases reconstruction and scene transfer on three JPEG photos. The broader prototype has rendered an 88-photo sample album, but automatic grouping, per-photo normalization, arbitrary natural-language interpretation, and professional-quality results on unseen albums remain unverified. The generated target is a deterministic test edit, not a generative model's output or a reconstruction of the human-edited dataset.
