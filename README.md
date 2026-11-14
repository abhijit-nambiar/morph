# Morph

Morph edits a whole album to match one or a few **reference edits**, consistently across every scene, and only touches **light, colour, tone, detail and noise** — it never crops, warps, moves, adds or removes anything.

You make the reference(s) one of two ways:

* **Your own edit** – spend your time on the one or few photos you like most (Lightroom, Capture One, anything) and hand Morph the exported JPEGs. Crops/straightening in your edit are fine; Morph registers it back onto the original.
* **Prompted** – describe the look in words, get several candidates from a generator (built-in local stand-in today; any image API drops in), pick one, then fine-tune global and per-region sliders (subject, background, sky, greenery, hair, face skin, body skin, eyes, lips, clothes) with a live preview.

Morph then reverse-engineers each reference into a deterministic recipe of Lightroom-style sliders, adapts it to every other photo's lighting, harmonizes within each scene and (more softly) across scenes, and exports full-resolution files with EXIF/ICC preserved.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'          # CPU-only is fine; models download from Hugging Face on first run
```

## Use

```bash
morph init  proj  path/to/album             # analyse photos (regions, faces, embeddings) + group into scenes
morph suggest proj                          # which photos make the best references, and why
morph studio  proj                          # web UI: prompt -> candidates -> pick -> region tweaks -> apply
```

Or entirely from the command line:

```bash
morph reference proj DSC_0906-4.jpg --edit my_edits/DSC_0906-4.jpg       # your own edit as a reference
morph generate  proj DSC_1280-55.jpg --prompt "warm golden hour, soft glowing skin, slightly darker background"
morph pick      proj DSC_1280-55.jpg 0                                     # choose candidate 0
morph adjust    proj DSC_1280-55.jpg face_skin.texture=-0.3 background.saturation=-0.2 eyes.exposure=0.15
morph apply     proj --out edited/                                         # whole album -> edited/photos, recipes, review page

# one shot
morph edit path/to/album --look "airy, lifted shadows, clean skin" --out edited/
morph edit path/to/album --edit-dir my_edits/ --out edited/               # use whichever photos you edited
```

`--generator` accepts `local`, `openai` (uses `OPENAI_API_KEY`, sends the original + prompt), any `https://…` endpoint that takes `{"image": b64png, "prompt", "n"}` and returns `{"images": [b64png…]}`, or `files:DIR` for candidates you made elsewhere.

Candidates from external generators (and uploaded hand edits) are re-segmented so the distortion guard can compare regions and face counts against the original; the built-in `local` generator only renders slider recipes, so it skips that step. Set `MORPH_THREADS=N` to cap CPU threads when several morph jobs share a machine.

## How it works

| Stage | Module | What it does |
|---|---|---|
| Analyse | `album.py`, `segment.py` | Working-res copy; soft masks for subject, background, sky, greenery, hair, face skin, body skin, eyes, lips, clothes (SegFormer human + face parsing + ADE scene parsing, edge-snapped with a guided filter); DINOv2 embeddings of the whole frame and of the background only; per-region Lab statistics, sharpness, clipping. Cached by file hash. |
| Scenes | `scenes.py` | Agglomerative clustering on background embedding + colour + capture time. |
| Suggest | `scenes.py` | Greedy coverage (facility location) over setting/lighting similarity, weighted by reference quality (sharp subject, faces, no clipping). |
| Generate | `generate.py`, `look.py` | Pluggable generators. The local one parses photographic vocabulary with region targeting and intensity modifiers ("slightly darker background", "very soft skin", "no grain") and renders variations on top of an auto-tone baseline. |
| Guard | `guard.py` | Every candidate is registered to the original (SIFT homography / ECC), then checked for edge overlap, local structure correlation, region-mask IoU and face count. Verdict: pass / repairable (changed pixels are excluded from fitting) / reject. |
| Fit | `fit.py`, `render.py` | A differentiable PyTorch renderer of global + per-region sliders (exposure, contrast, highlights, shadows, whites, blacks, WB, vibrance/saturation, HSL ×8, colour grading, curves, clarity, texture, dehaze, noise reduction, sharpening, grain, vignette); Adam fits it to the target in Lab, confidence-weighted and region-balanced. |
| Propagate | `propagate.py` | Blends reference recipes by similarity; predicts where each region should land (reference result + damped share of this photo's difference); harmonizes those targets within a scene and across scenes; solves a small per-photo correction so rendered region statistics hit them. |
| Export | `export.py` | Full-res render with guided-upsampled masks, EXIF/ICC/XMP kept, geometry + edge check on every file, per-photo recipe JSON, before/after review page. |

Benchmark against a hand-edited album (use K of your edits as references, predict the rest):

```bash
morph evaluate Morph/originals Morph/edits --out runs/eval -k 3
```

On the bundled 88-photo album (mean CIEDE2000 to the hand edit on held-out photos; lower is better):

| references | untouched original | copy nearest reference's recipe | morph |
|---|---|---|---|
| 3 | 8.72 | 5.93 | **5.23** |
| 6 | 8.73 | 4.72 | **4.55** |

## Research archive

Everything below documents the earlier research prototype and static showcase, kept for reference.

Morph learns a look from one edited reference photo, corrects small lighting/settings differences, and reuses the saved look across a scene. The showcase contains 88 photos, 14 reviewed lighting groups and three styles, with an eight-stage interactive pipeline walkthrough.

## Run the showcase after cloning

The saved website is static HTML, CSS, JavaScript and images. Serving it requires **Git, Git LFS, Python 3 and a browser**. Python 3.11 is recommended. You do not need Node.js, npm, API keys, a GPU or the image-processing dependencies to view it.

Install [Git LFS](https://git-lfs.com/) before cloning. On macOS with Homebrew, use `brew install git-lfs`; on Ubuntu/Debian, use `sudo apt install git-lfs`. Then:

```bash
git clone https://github.com/abhijit-nambiar/morph.git
cd morph
git lfs install --local
git lfs pull
python3 -m http.server 8765 --bind 127.0.0.1 --directory outputs/morph-showcase
```

Open **[the pipeline walkthrough](http://127.0.0.1:8765/#process)** or [the cover](http://127.0.0.1:8765/). Keep the terminal running while presenting; press **Ctrl+C** to stop the server. The server serves only the showcase folder on your own machine. Once the clone and LFS downloads finish, the saved showcase works offline.

On Windows, install Git LFS and Python 3, run the same Git commands, then use:

```powershell
py -3 -m http.server 8765 --bind 127.0.0.1 --directory outputs/morph-showcase
```

Run these commands from the **repository root**, the folder containing this README. The showcase output folder is approximately **3 GB**; allow additional space for the dataset, historical outputs and Git LFS objects.

### What must be present

Keep the entire [showcase folder](./outputs/morph-showcase/) together. It contains:

- `index.html` and `showcase-data.json`: the page and its saved data. The page embeds its data and does not run the image pipeline in the browser.
- `assets/`: preview images for comparisons, the walkthrough and gallery.
- `models/`, `varied-album/` and `final-albums/`: linked fitting evidence, simulated inputs and full-resolution exports.
- `local-experiment/`: the separately labeled person-aware example.
- The JSON reports and Markdown notes linked from the page.

Website links are relative to `outputs/morph-showcase/`. File references in the saved current showcase and attached person-experiment reports are relative to the **repository root**. Keep the repository structure intact; no particular username or checkout location is required. Original images are in [Morph/originals](./Morph/originals/), with 88 corresponding human edits in [Morph/edits](./Morph/edits/).

**Before cloning elsewhere:** the latest README, [implementation](./outputs/morph-starter/) and complete showcase folder must be committed and pushed to the branch you clone. Uncommitted local files are not included in a clone. The repository's `.gitattributes` uses Git LFS for images, so their LFS objects must also be uploaded.

### Presenting

Start with **Pipeline walkthrough**, choose a scene/photo/style, and follow **Group → Style → Fit → Vary → Normalize → Replay → Album → Verify**. Each step shows saved inputs, outputs and evidence. Use the step tabs or Previous/Next; arrow keys, Home and End navigate a focused step tab.

**Present** enables chapter navigation with left/right arrows; Escape exits. The album gallery supports scene filtering and a five-stage photo inspector. See [presenter notes](./outputs/morph-showcase/PRESENTER-NOTES.md) for a five-to-seven-minute walkthrough.

The three 88-photo styled albums use **global tone/color and per-photo normalization**. Face-region and person-mask lighting are separate experiments, illustrated in the local-light chapter; they are not applied to those albums.

### Troubleshooting

- **Missing `index.html`:** confirm you are at the repo root and have the branch containing `outputs/morph-showcase/`. Running a server cannot recover outputs that were never committed/pushed.
- **Images are missing or tiny text files:** run `git lfs pull`. An LFS pointer starts with `version https://git-lfs.github.com/spec/v1`; the actual image content must be downloaded. Resolve any reported Git/LFS authentication error before continuing.
- **Port already in use:** replace `8765` with `8766` in the server command and browser URL.
- **`python3` is not found:** install Python 3, or use `python`/`py -3` if that is the Python 3 command on your system.
- **Old page after updating the repo:** reload the browser. If serving from another working directory, use the correct path to `outputs/morph-showcase` in `--directory`.

## Optional: regenerate the showcase

Serving the saved website does not require regeneration. To rerun fitting, simulated variation, normalization, exports and evaluation, install the processing dependencies into a virtual environment. Run from the repo root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r outputs/morph-starter/requirements-normalization.txt
.venv/bin/python outputs/morph-starter/showcase_workflow.py Morph/originals \
  --scenes outputs/morph-starter/showcase-scenes.json \
  --styles outputs/morph-starter/showcase-styles.json \
  --seed 20261030 --jobs 1 --out outputs/morph-showcase-reproduction
python3 -m http.server 8766 --bind 127.0.0.1 --directory outputs/morph-showcase-reproduction
```

On Windows, use `py -3 -m venv .venv` and `.venv\Scripts\python.exe` for the processing commands; join the continued command onto one line in PowerShell.

Choose a **new output directory** for each run. Full-resolution processing can take several minutes and needs additional disk space. The workflow selects the renderer using validation; it stops for inspection if cubic is not selected. Increase `--jobs` to `2` or `3` if memory permits. The source originals and human edits are preserved.

To include the same historical person-mask example, append `--person-experiment outputs/morph-resumed` to the workflow command. This copies an existing mask/result; generating new person masks separately requires macOS Vision and Swift command-line tools. The default reproduction explains the local-light branch without attaching its historical images.

To check a downloaded showcase's preview images and file links after installing the dependencies:

```bash
.venv/bin/python outputs/morph-starter/verify_showcase_artifact.py outputs/morph-showcase \
  --out showcase-local-verification.json
```

The verification output must be a new file. To run the portable end-to-end workflow tests:

```bash
.venv/bin/python -m unittest discover -s outputs/morph-starter -p 'test_style_workflow.py'
```

The full implementation and additional research commands are documented in [the starter README](./outputs/morph-starter/README.md). The saved site was also checked from a relocated repository copy: all 1,334 linked website files resolved and all 306 full-resolution exports verified. See [the portability report](./outputs/morph-showcase/portability-verification.json).

## Results and research history

The [results report](./outputs/morph-showcase/RESULTS.md) records measurements, verification and limitations. This is a controlled demonstration on one reviewed album using synthetic settings variations and agent-authored style targets. It measures restoration against a reference-calibrated clean baseline. It does not establish human-edit quality or unseen-album performance. Across-scene intent depends on intended references and a shared style recipe.

- [Current showcase notes](./outputs/morph-showcase/README.md): output layout and interpretation.
- [Person-light experiment](./outputs/morph-resumed/): the separate 13-photo, two-scene experiment and mask reviews.
- [Code-only ZIP](./outputs/morph-starter.zip): reusable implementation, without the showcase images.
- [Earlier reconstruction](./outputs/morph-reconstruction/) and [research presenter notes](./outputs/morph-research-presenter-notes.md): historical research. Archived logs/reports may retain paths from their original machines; they are not dependencies of the current site.
- [Session 27](./27/) and [session 28](./28/): historical agent/approval logs.

Next research priorities are aligned evaluation of the 88 supplied human edits, additional albums with photographer review, reliable grouping/reference selection, and local illumination/subject-mask validation. No CI or deployment is configured for this local showcase.
