# Morph: scene edit reconstruction starter

This is a local command-line research prototype. It proposes photo groups, generates deterministic color-and-tone targets, fits a reproducible transformation, and transfers it to each group. It does not need an API key or alter image geometry. An optional experimental path estimates global photo corrections from matched scene content and designated references. It does not yet understand free-form text or establish professional quality. Review the outputs and proposed groups.

## Styled album and normalization showcase

`showcase_workflow.py` reproduces an offline showcase with three agent-authored looks, simulated settings differences, reference-based normalization, full-resolution styled JPEGs, evaluation and file verification. From the repository root:

```bash
.venv/bin/python -m pip install -r outputs/morph-starter/requirements-normalization.txt
.venv/bin/python outputs/morph-starter/showcase_workflow.py Morph/originals \
  --scenes outputs/morph-starter/showcase-scenes.json \
  --styles outputs/morph-starter/showcase-styles.json \
  --seed 20261030 --jobs 3 --out outputs/morph-showcase-reproduction
```

Use a fresh output folder. The supplied scene plan covers this 88-photo album; replace it with reviewed groups for another album. The command checks exact photo coverage, records stages in `run.json`, and writes logs under `logs/`. Three concurrent full-resolution style renders require substantial memory; use `--jobs 1` on smaller machines. Open the resulting `index.html` locally. It includes an eight-stage guided pipeline with scene/photo/style selection, real fitting previews and correction values, style switches, scene filters, processing-stage inspection, presentation controls and full-resolution links, with no external assets. The local-light section distinguishes global album processing from the separate experimental face/person paths. Add `--person-experiment outputs/morph-resumed` to include a labeled mask/overlay/result example from an existing verified person run; it does not apply that local edit to the styled album.

The independent target editor uses Pillow tone curves and HSV color adjustments. A benchmark compares existing renderers with an optional cubic RGB mapping, selects by validation frames, and fits one representative per lighting group. This showcase expects cubic to win; if another renderer wins, it stops for inspection. Cubic is also available through `morph.py fit` and `album` with `--renderer cubic --solver least-squares --regularization .000001`. Existing renderer defaults are unchanged.

`make_variances.py` saves new copies with bounded scene-wide and per-photo exposure-like, white-balance and contrast shifts. These are global logit tone curves, not physical camera exposure stops or local relighting. `photometric_normalization.py` receives only those copies and one designated intended reference per group. It finds SIFT correspondences, verifies geometry, and robustly estimates global tone/color corrections across a graph of overlapping frames. Geometric alignment is used only for measurement; exported framing is unchanged. Unconnected or out-of-bounds estimates are flagged for review and receive identity corrections. The resulting photo recipes use the existing correction mechanism before the saved scene style.

Each reference defines its scene's absolute exposure/color. A shared style recipe provides album intent, but the pipeline does not independently choose an ideal relationship between disconnected scenes. Select references with that intent in mind. The single shaded close-up has its own reference and supplies no within-scene transfer evidence.

`evaluate_style_showcase.py` separately reads withheld clean counterparts and injected settings. Its intended baseline is the same normalizer run on the unperturbed album; this tests stability under injected differences, not independent aesthetic correctness. It reports photo error, per-scene residual bias and across-scene bias spread, plus decoded styled JPEG error and a separate noncalibration summary. The synthetic variance family matches the corrector's global model, making this a favorable controlled test. Human-edited counterparts, unseen albums, clipping recovery and local illumination require separate validation.

`verify_style_showcase.py` checks source hashes, coverage, decoding, original oriented dimensions and ICC/EXIF presence for all 264 album exports plus 42 fitting-reference exports. `verify_showcase_artifact.py` checks every preview and local link. Browser layout and interaction QA remain a separate check. For the actual album's measurements and verification, see `../morph-showcase/RESULTS.md`.

## Setup

Requires Python 3, NumPy and Pillow. In a terminal from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy Pillow
python test_pipeline.py
```

The tests use temporary synthetic images; they do not need the original dataset. This package does not contain your photos or derived images. Files and paths below are examples: replace `/path/to/originals` with your own album folder.

## One-command prompt workflow

Start with an originals folder for a chronological grouping proposal, or replace `--originals /path/to/originals` with a reviewed group JSON path:

```bash
python album_workflow.py --originals /path/to/originals \
  --prompt 'warm and slightly soft contrast and lift shadows' --out album-preview
```

Global color and tone phrases use the gamut renderer without face detection. A phrase such as `brighten upper left` uses the experimental frame-relative grid. A face-light phrase such as `brighten faces` requires `--detections detections.json` or `--detector ./face-detector` and uses the experimental macOS face workflow described below. Run `python morph.py prompt-help` for the supported vocabulary; free-form requests are not understood.

Some common equivalents are accepted, including `warmer`, `open shadows`, `reduce highlights`, `increase exposure`, and `reduce contrast`. They map to the same target controls as the canonical phrases listed by `prompt-help`; unsupported wording and repeated or conflicting controls are rejected. The original prompt and canonical interpretation are saved with the fit.

The default renders a few full-resolution samples per proposed scene. Open `album-preview/review/index.html` and inspect `groups-review`; chronological groups are a heuristic and the command does not pause for review. Add `--full` to render the entire album and check its files. Use a new output folder each time. The workflow records completed stages in `run.json`; a failed run can leave partial output. It does not guarantee professional quality or automatic normalization. Run `python test_album_workflow.py` for route, export and failure checks.

## Run an album

### Experimental person-light reconstruction

For a single reviewed scene on macOS, a separate experimental path can reconstruct a synthetic `brighten people` or `darken people` effect and replay it across that scene:

```bash
swiftc person_segmentation.swift -o person-segmentation
python3 person_masks.py reviewed-groups.json --scene scene-01 --extractor ./person-segmentation --out person-masks
python3 review_person_masks.py person-masks/manifest.json --out person-mask-review
python3 person_scene.py fit reviewed-groups.json person-masks/manifest.json --scene scene-01 --prompt 'warm and lift shadows and brighten people' --out person-fit
python3 person_scene.py apply person-fit/fit.json --preview-photo example.jpg --out person-preview
python3 review_person_scene.py person-preview/report.json --out person-scene-review
```

Replace `example.jpg` with an actual scene filename; repeat `--preview-photo` for more photos, or omit it to render the scene. Open `person-mask-review/index.html` to inspect thumbnail overlays before fitting, then use `person-scene-review/index.html` for side-by-side thumbnails and links to the full-resolution exports. The scene review checks dimensions and reports ICC presence; it does not establish quality or detect clipping. Masks are low resolution, may miss people or include background, and are interpolated to export dimensions. They affect clothing as well as skin and do not identify individual people. An empty-mask flag detects almost-empty masks only; a partially detected group can pass it, so inspect every overlay. The mask manifest binds each mask to the source and verifies hashes before fitting and replay; editing a mask requires a new matching manifest with its updated hash. Empty masks are flagged on export, while a representative with an empty mask is rejected. This is not automatic normalization or a validated real-edit model. Synthetic targets and reconstruction share masks. Run `python3 test_person_scene.py` to exercise fit, replay, review, and mask-integrity checks.

If distant people are missed, an experimental multi-scale pass can combine whole-frame and overlapping-region detections: `python3 tiled_person_masks_experiment.py person-masks/manifest.json --extractor ./person-segmentation --out person-multiscale-masks`. Review the new manifest with `review_person_masks.py` and pass it to `person_scene.py fit` if suitable. Detection regions do not alter the exported photo dimensions. The combined mask may still miss people or add false positives and needs visual review; it takes five segmentation calls per image. Run `python3 test_tiled_person_masks.py` for the assembly and integrity checks.

To use a manually corrected mask or select a different generated mask for specific photos, save the replacements as separate 8-bit grayscale PNGs without modifying the originals. Run `python3 replace_person_masks.py person-masks/manifest.json --replace example.jpg /path/to/replacement.png --out selected-masks`; repeat `--replace` for additional photos. The command copies all masks into a new directory and writes a new checksum manifest. It does not verify that a replacement is aligned with the named photo, so inspect `selected-masks/manifest.json` with `review_person_masks.py` before fitting. It rejects changed source files, changed original masks, duplicate choices, and unknown filenames. Run `python3 test_replace_person_masks.py` to exercise these checks.

The person-scene `apply` command also accepts `--corrections corrections.json` using the album, scene, and photo structure in the manual corrections example below. For example, `{"photos":{"scene-01":{"example.jpg":{"exposure":0.2}}}}` adjusts just that photo. Corrections are applied in order: photo correction, fitted person-aware look, scene correction, album correction. Use the current scene ID in the file and preview full-resolution outputs before rendering the entire scene; this is manual adjustment, not automatic normalization. A per-scene apply rejects correction entries for other scenes.

For several scenes, combine their mask manifest lists into one JSON list, then fit and preview one mapping per scene:

```bash
python3 person_album.py fit reviewed-groups.json all-masks.json \
  --prompt 'warm and slightly soft contrast and lift shadows and brighten people' --out person-album-fits
python3 person_album.py apply person-album-fits --preview-samples 2 --out person-album-preview
python3 review_person_album.py person-album-preview --out person-album-review
```

Open `person-album-review/index.html` for comparisons grouped by scene and links to full-resolution exports. Omit `--preview-samples` to render every photo into a new output folder. Album replay accepts the same `--corrections` file and validates its scene/photo assignments. The review checks reported export coverage, image decodability and oriented dimensions, reports ICC presence, and rejects incomplete output. It does not validate mask accuracy, professional quality or automatic normalization. Run `python3 test_person_album.py` for multi-scene preview, replay, correction routing, review links and failure checks. All saved input paths must exist on the current machine; moved albums require regenerated path manifests and refitting.

On macOS, an experimental content-aware grouping proposal is also available. It splits chronological groups using local Vision image-feature distances; it does not perform segmentation, cannot reunite scenes across chronological boundaries, and the default threshold is unvalidated. It can split similar photos too finely. Compile the helper once, then run:

```bash
swiftc vision_scene_features.swift -o vision-scene-features
python3 vision_group.py /path/to/originals --extractor ./vision-scene-features --out vision-groups
```

Open `vision-groups/review/index.html` for thumbnail contact sheets, and correct `vision-groups/groups.json` using `edit_scene_groups.py` as described below. Pass the reviewed group file to `album_workflow.py` as its positional argument. The grouping run records SHA-256 source hashes and checks that files do not change during extraction and sheet generation; later workflows do not automatically recheck that particular manifest. Fitting records its own source checksums. Vision requires local macOS image services and may fail in a restricted sandbox; no images are sent to a remote service. The pairwise distance matrix is limited to 1,000 photos and may be slow for large albums. Run `python3 test_vision_group.py` for a synthetic grouping-to-preview check.

```bash
python morph.py group /path/to/originals --method chronological --gap-minutes 5 --out groups.json
python review_groups.py groups.json --source /path/to/originals --out groups-review
```

Inspect the contact sheets and edit `groups.json` if needed. For an editable table, run `python edit_scene_groups.py build groups.json --out assignments.csv`, change each row’s `scene`, and mark exactly one representative per scene with `yes`. Compile with `python edit_scene_groups.py compile groups.json assignments.csv --out reviewed-groups.json`. The compiler rejects missing, duplicate, unknown, or invalid assignments and preserves every original photo. Refit after changing groups; existing fits belong to their original memberships. A red border marks each group's representative. Groups use capture time/camera model and do not reliably recognize scenes. Images without capture times become single-photo groups. `groups-review/index.json` lists unassigned images.

```bash
python morph.py album groups.json \
  --target-engine pillow-soft --target-settings example-target-settings.json \
  --renderer gamut --solver gauss-newton --full-res-reference --out album-output
python review_album.py album-output/review.json --out album-review
```

This creates a deterministic target from each representative using independent per-channel tone curves and color-range adjustments, fits the bounded renderer, and applies its recovered settings to each original in the group. The single target settings file defines a shared test look; edit it to experiment with different combinations. Target controls and renderer controls have different meanings.

To review representatives before processing the whole album, add `--preview-only --full-res-reference` to the `album` command and use a new output folder. Add `--preview-samples 3` to render up to three full-resolution photos per scene in a `samples/` folder, with per-sample clipping diagnostics. The first is the representative; others are selected by simple appearance diversity. Preview mode does not export or check the rest of the album. Then run `python morph.py apply groups.json PREVIEW_FOLDER --out album-output` to render all originals from those fits.

Add `--evaluate-transfer` to compare each rendered sample at fitting resolution with an independently generated target for that photo. `review.json` records the source and output errors; `report.json` counts evaluated non-representative photos whose output is worse than the source. With supplied targets, missing or differently sized targets are skipped; same-size pairs may still be misaligned or have different intended edits. The metric is pre-JPEG pixel error, not a photographic-quality score, and preview mode covers only selected samples.

As an alternative to `--target-settings`, use a small offline prompt vocabulary: `--prompt 'warm, slightly soft contrast and lift shadows'`. Run `python morph.py prompt-help` to list supported phrases. It rejects unsupported or conflicting instructions; it is a deterministic rule-based target stub, not general language understanding or a generative model. The prompt and parsed target controls are saved in each `fit.json`.

After grouping, `python prompt_semantics.py groups.json --out prompt-semantics.json` runs numerical directional sanity checks on each phrase and one transferred photo per group. It is not a visual-quality or natural-language evaluation.

Each scene has `fit.json` with its target settings, recovered and applied recipe, and fit metrics. `target.jpg` and `reconstruction.jpg` are small previews used for fitting. `target-full.jpg` and `reconstruction-full.jpg` are full-resolution representative comparisons when `--full-res-reference` is used. Each `exports/` folder contains full-resolution edited photos. Source EXIF is preserved where readable, orientation is normalized, and output uses an sRGB ICC profile; other metadata such as XMP/IPTC may not be preserved.

`review.json` flags changes in near-black or near-white pixels before JPEG encoding. A flag is a prompt to inspect a photo, not proof of bad quality; no flag is not proof of good quality. The contact sheets are thumbnails: open the individual JPEGs for full-resolution review.

For polynomial and grid mappings, `review.json` also reports the sampled fraction of pixels whose mapped channels fell outside 0–1 before clipping. The grid value is `grid_out_of_gamut_fraction` and uses original image coordinates when sampling. With corrections enabled, this is measured after the per-photo correction and before later scene and album corrections.

## Replay or refine a look

```bash
python morph.py apply groups.json album-output --out album-replay
```

This uses the stored fitted settings and renders from originals without fitting again. Add `--strength 0.7` to scale the recovered look toward neutral. New fits include source-file SHA-256 checksums and scene membership. `apply` rejects changed photos or group membership unless you explicitly pass `--allow-changed-inputs`; a changed representative is always rejected. Older fits without checksums work with a warning and are marked `legacy-unverified`.

For a saved grid fit, `python3 morph.py apply groups.json FITS --local-strength 0.5 --preview-samples 3 --out reduced-local-preview` scales only its local-light coefficients between 0 and 1; `--strength` still scales the entire mapping first. Omit `--local-strength` to preserve the full local contribution. The value is recorded in `applied.json`. To set a different value per scene, add `"local_strengths": {"scene-01": 0.5, "scene-02": 0.0}` to the corrections JSON; these values override `--local-strength` for those scenes, while unspecified grid scenes use the command-line value or 1.0. Unknown scenes, non-grid scenes with overrides, and values outside 0–1 are rejected before rendering. This option requires every scene fit in the batch to use the grid renderer. The global and local coefficients were fitted jointly, so zero local strength does not reproduce a separately fitted global-only model. Review the full-resolution samples before applying the choice to a whole album.

For a thumbnail comparison at local strengths 0, 0.5 and 1, run `python3 review_local_light.py build groups.json FITS --out local-review`. Enter one value from 0–1 per reviewed scene in `local-review/decisions.csv`, then run `python3 review_local_light.py compile local-review/decisions.csv --out local-corrections.json`. Blank scenes are omitted; `apply --corrections local-corrections.json` therefore uses its usual default for those scenes. The comparison shows only selected photos and does not include other correction layers or check checksums; `apply` validates source checksums where available. If combining with other corrections, merge the `local_strengths` object into your existing corrections JSON. Export full-resolution samples with `apply --preview-samples 3` before the whole album.

To set intentional corrections, create a JSON file like:

```json
{
  "album": {"temperature": 0.02},
  "scenes": {"scene-01": {"shadows": 0.06}},
  "photos": {"scene-01": {"photo.jpg": {"exposure": 0.08}}}
}
```

Replace scene IDs and filenames with ones in your group file, then add `--corrections corrections.json` to `apply`. Operations are composed in order: per-photo correction, fitted scene look, scene correction, album correction. Corrections are manual and use the saved scene renderer's control definitions; polynomial, spatial and grid mappings use the gamut renderer for the corrections. Each output scene records its applied recipe and corrections in `applied.json`.

For gamut, polynomial, spatial, or grid fits, `python suggest_exposure.py groups.json ALBUM_FITS --out draft-corrections.json` can draft per-photo exposure candidates from capture metadata. **Review them before applying:** this heuristic had mixed results on one album and does not know whether scene light or intent changed. Its `.audit.json` records the candidates and exclusions. To inspect up to three full-resolution photos per scene, prioritizing individually corrected photos, run `python morph.py apply groups.json ALBUM_FITS --corrections draft-corrections.json --preview-samples 3 --out correction-preview`. This does not render or check the rest of the album. Edit or remove suggestions before producing the full album.

For an individual review list, run `python review_exposure.py build groups.json ALBUM_FITS draft-corrections.json.audit.json --out exposure-review`. The resulting contact sheets are thumbnails; they show original, scene look, and suggested correction. In `decisions.csv`, enter `accept` or `reject` per candidate. Then `python review_exposure.py compile exposure-review/decisions.csv --out approved-corrections.json` writes only accepted corrections; blank entries are excluded. Use the approved file with `apply`. Review full-resolution samples as needed; thumbnail review itself does not verify source checksums, while `apply` does when the fit has them.

For a deterministic off-center test target, use `--target-engine gaussian-soft --prompt 'warm and lift shadows' --target-local-light .3 --target-light-center .2 .3 --target-light-radius .18 .23 --renderer grid --solver least-squares --regularization .0001`. Center and radius are fractions of frame width and height (center range 0–1; radius range 0.03–1). The target combines Pillow color/tone adjustments with a smooth Gaussian light region; its settings are saved in `fit.json`. The mask is fixed to image coordinates and does not follow subjects. `--full-res-reference` and `--evaluate-transfer` work for this target. With `gaussian-soft`, a prompt can also include one frame-relative phrase such as `brighten upper left`, `slightly darken center`, or `strongly brighten lower right`. Run `prompt-help` for supported region names. A local phrase cannot be combined with an explicit local strength or center; a radius may still be supplied. The parser does not recognize subjects or arbitrary directions.

## One-command face-album preview

Once scene groups and checksum-verified face detections have been reviewed, run:

```bash
python face_album_workflow.py reviewed-groups.json --detections detections.json \
  --prompt 'warm and lift shadows and slightly brighten faces' --out face-run
```

You can start from a folder instead of a group file by replacing `reviewed-groups.json` with `--originals /path/to/originals`. This proposes chronological groups using capture time and camera; it is not scene recognition, and images without capture times become singleton scenes. The run proceeds without pausing for group review, so inspect `groups-review` and refit with a corrected group file before relying on the result. This creates full-resolution preview samples, detection sheets, scene fits, an album review index, and sampled clipping diagnostics. Open `face-run/review/index.html`. Add `--full` to render the entire album and run file verification; use a new output folder. You may provide `--detector ./face-detector` instead of `--detections` to run local detection first, but inspect the generated boxes: automatic detection is imperfect and the script does not pause for approval. The supplied group file should be reviewed before use. The face workflow is experimental and the prompt language is limited. Failed runs leave partial outputs and a `run.json` status; start again in a new folder after addressing the error. Run `python test_face_album_workflow.py` for an end-to-end synthetic test.

## Experimental face-relative lighting (macOS)

This separate scene workflow uses the local macOS Vision framework and needs the Swift command-line tools. It applies a smooth region around every detected face, including nearby background; it does not select or track particular people. It has not been validated for professional quality. Review the detections and full-resolution samples.

```bash
mkdir -p .swift-cache
swiftc -module-cache-path .swift-cache detect_faces.swift -o face-detector
python detect_scene_faces.py groups.json --detector ./face-detector --out detections.json
python review_face_detections.py detections.json --out detection-review
python face_scene_pipeline.py fit groups.json detections.json --scene scene-01 \
  --prompt 'warm and brighten faces' --out face-fit
python face_scene_pipeline.py apply face-fit/fit.json --preview-samples 3 --out face-preview
```

Inspect the thumbnail sheets in `detection-review` and edit `detections.json` if needed before fitting: source checksums prove file identity, not correct detections. To reconstruct supplied edits, use `--targets /path/to/edits` instead of `--prompt` with `fit` or `fit-album`. The folder must contain a same-name, same-size edit for each representative. Equal dimensions do not guarantee that original and edit are aligned or that their edits are consistent; inspect pairs. The fit records the target path and checksum. If any photo in a scene has a face detection, the representative must have one too. If no photos in a scene have detections, only the global mapping is fitted; the scene is flagged for review. Photos with no detections receive the recovered global mapping and are flagged in `report.json`; detector errors stop fitting. The compiler and detector run locally; in restricted environments, macOS image services may need to run outside the sandbox. Once reviewed, omit `--preview-samples` to render that scene; use a new output folder. For all reviewed scenes, replace the single-scene commands with `python face_scene_pipeline.py fit-album groups.json detections.json --prompt "warm and brighten faces" --out face-album-fits`, then `python face_scene_pipeline.py apply-album face-album-fits --preview-samples 3 --out face-album-preview`. Omit `--preview-samples` for a full album. Representatives need a face detection when their scene has detected faces; all scene inputs are checked before rendering begins. An unexpected error during rendering may leave partial outputs. Run `python test_face_scene_pipeline.py` for synthetic integrity and replay checks.

For a thumbnail overview of face-album output, run `python3 review_face_album.py ALBUM_OUTPUT --out album-review`. `album-review/index.json` maps each original to its full-resolution export; the sheets themselves are previews. Photos without detected faces are flagged for inspection.

Face-scene exports record reduced-resolution, pre-JPEG clipping diagnostics in each scene report. `near_limit_flag` means near-black or near-white pixel fraction increased by more than one percentage point; `face_mapping_out_of_gamut_fraction` measures pixels outside the display range before the fitted mapping clamps them. These are sampled heuristics, not visual-quality verdicts, and may miss small damaged areas. For an older export, run `python3 audit_face_album.py FACE_ALBUM_FITS ALBUM_OUTPUT --out clipping-audit.json` to recompute them from the originals and stored corrections.

After a full face album export, run `python3 verify_face_album.py FACE_ALBUM_FITS ALBUM_OUTPUT --out verification.json` to check photo coverage, original checksums, decoded dimensions, and ICC/EXIF presence. It flags images without face detections and missing or unexpected outputs. It does not check image quality, correct detection, all metadata, or every processing artifact.

For small faces missed by the whole-frame detector, an optional `detect_faces_regions.swift` scans overlapping image regions locally and returns boxes in original image coordinates; the photos themselves are unchanged. Compile it like the first detector, run `detect_scene_faces.py` on a group file containing only the photos needing a second pass, then use `python3 merge_face_detections.py detections.json second-pass.json --out merged-detections.json`. The merge replaces only previously empty detections and verifies checksums. Run `review_face_detections.py` on the merged file and inspect the boxes: some people may still be missed, and false detections are possible. This option was tested on four photos in one album, not validated generally.

`python3 evaluate_face_transfer.py FACE_ALBUM_FITS FACE_ALBUM_OUTPUT --out transfer.json` checks rendered photos against their independently recomputed synthetic targets at fitting resolution. It excludes representatives from the held-out summary and applies the stored correction layers. The target and fitted model use the same detected face boxes, and the score is pre-JPEG pixel error, not a photographic-quality or human-edit score. Preview runs only evaluate their selected samples. The face one-command workflow writes `transfer-evaluation.json` automatically.

`audit_scene_consistency.py FACE_ALBUM_FITS ALBUM_OUTPUT --out scene-consistency.json` is a review-only brightness check. It compares Gaussian-weighted luminance around detected face boxes to the median within each scene and flags large log2 luminance ratios. At least three measured photos are needed per scene; previews may contain too few. The values are computed on reduced-resolution, gamma-encoded images, not calibrated camera exposure stops. Face boxes can miss people or include surroundings, and skin tone, composition and intent can legitimately differ. The output suggests photos to inspect and applies no correction. The face one-command workflow produces this report automatically.

To review the flagged photos, run `python3 review_scene_consistency.py build FACE_ALBUM_FITS ALBUM_OUTPUT scene-consistency.json --out consistency-review`. The contact sheets show the current export and two exposure alternatives as thumbnails. They are examples, not recommendations; this review requires an album rendered without correction layers. Enter a numeric exposure between -0.5 and 0.5 in `consistency-review/decisions.csv` only for photos you want to adjust, then run `python3 review_scene_consistency.py compile consistency-review/decisions.csv --out reviewed-corrections.json`. Blank entries and zero add no correction. These exposure units are renderer parameters, not measured camera stops. Use `--corrections reviewed-corrections.json` with `face_scene_pipeline.py apply` and `--preview-photo` to inspect selected photos at full resolution before exporting an album. The contact sheets are reduced-resolution comparisons and do not establish artistic quality.

To inspect a particular full-resolution photo in one fitted face scene, use `python3 face_scene_pipeline.py apply FIT/fit.json --preview-photo PHOTO_FILENAME --out photo-preview`. Repeat `--preview-photo` for more than one scene member. It cannot be combined with `--preview-samples`; invalid or duplicate filenames are rejected. Add `--corrections` to compare reviewed adjustments.

The face workflow also accepts `--corrections corrections.json` on `apply` and `apply-album`, using the same `album`, `scenes`, and `photos` structure shown above. Corrections use the gamut renderer and are composed in this order: photo correction, fitted face look, scene correction, album correction. Unknown scene or photo names are rejected before batch rendering. In preview mode, the representative and individually corrected photos are prioritized. These are manual adjustments, not automatic exposure normalization; inspect full-resolution samples.

For review-only metadata suggestions with a face album fit, run `python3 suggest_face_exposure.py FACE_ALBUM_FITS --out draft-face-corrections.json`, then `python3 review_face_exposure.py FACE_ALBUM_FITS draft-face-corrections.json.audit.json --out face-exposure-review`. The sheets are thumbnails showing original, fitted look, and suggestion; other correction layers are not shown. Enter `accept` or `reject` in the decisions CSV, then use `python3 review_exposure.py compile face-exposure-review/decisions.csv --out approved-face-corrections.json`. Blank decisions are excluded. Use that file with `face_scene_pipeline.py apply-album --corrections` and render full-resolution samples before the album. Merge the resulting `photos` object into other corrections if needed. The metadata heuristic is imperfect and has had mixed results on one album; suggestions are not automatically applied.

## Other target and mapping options

- Fit one existing same-sized, aligned representative pair with `python morph.py fit ORIGINAL.jpg TARGET.jpg --renderer gamut --solver gauss-newton --out fit-output`.
- Use existing same-name, same-size representative targets across the groups with `python morph.py album groups.json --targets /path/to/edits --renderer gamut --solver gauss-newton --out album-from-targets`. Same dimensions do not guarantee alignment; inspect the pairs.
- `--renderer polynomial --solver least-squares --regularization .001` fits a deterministic nonlinear RGB color mapping rather than slider controls. It is global and can clip or overfit; compare its results with the bounded renderer.
- `--renderer spatial --solver least-squares --regularization .0001` adds a fixed center-weighted local-light component to a global color map. Pair it with `--target-engine spatial-soft --target-local-light .3` for an independent deterministic test target. This mask is tied to the frame, not the subject, and may transfer poorly when composition changes. It is experimental; assess samples before batch export. Corrections use the gamut renderer.
- `--renderer grid --solver least-squares --regularization .0001` fits a global color mapping plus nine smooth local-light regions. It is experimental: regions are tied to the frame rather than subjects, and the mapping can overfit or clip. Inspect full-resolution scene samples before export; corrections use the gamut renderer.
- `--target-engine pillow-selective` creates a stronger target with independent global and hue-range operations. It can introduce clipping. The current `--full-res-reference` option is supported for `pillow-soft`, `spatial-soft`, `gaussian-soft`, `morph`, and supplied targets, not the other Pillow target engines.

`group`, `target`, `fit`, `album`, `apply`, and `evaluate` refuse to overwrite their inputs or existing planned outputs. Use a new output directory, or pass `--overwrite` to replace planned outputs (this never allows overwriting an input and does not delete stale files). Full-resolution processing is sequential and can take minutes for a large album.
