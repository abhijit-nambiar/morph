# Morph showcase results

The controlled 88-photo album run is complete. All 88 normalization estimates were accepted; no photo was left uncorrected. All 264 styled varied-photo outputs and 42 fitting-reference exports passed independent file verification. Original photos and supplied human edits are unchanged.

## Final measurements

RGB RMSE uses a 0–1 range. Normalization is evaluated at maximum side 1000 pixels. Styled results are measured from decoded full-resolution JPEG exports resized for evaluation. The before column applies the saved style to a varied input without correction; the after column uses normalization followed by that style.

| Stage / look | Before | After | Reduction | Improved |
|---|---:|---:|---:|---:|
| Normalization | 0.04159 | 0.00431 | 89.6% | 88/88 |
| Sunlit editorial | 0.04076 | 0.00705 | 82.7% | 88/88 |
| Coastal cool | 0.04240 | 0.00610 | 85.6% | 88/88 |
| Soft film | 0.03682 | 0.00783 | 78.7% | 88/88 |

Excluding the 14 designated-reference counterparts, all 74 remaining photos improved in normalization and in each styled export. Their mean normalization error was 0.04588 → 0.00454. These photos were excluded from style fitting, but this album was used during method development.

RMS of the per-scene residual RGB biases fell from 0.02046 to 0.00136; the across-scene spread of those biases fell from 0.02033 to 0.00125. Both measure restoration relative to the reference-calibrated clean baseline, not uniform brightness of different scenes.

Full per-photo, per-scene and noncalibration breakdowns are in `evaluation.json`. The largest styled residuals are in the lawn group; the worst mean-per-photo RMSE is 0.02133 for Soft film on `DSC_0947-13.jpg`. Inspect that example alongside the strongest images.

## Style reconstruction benchmark

Each style has 14 fitting representatives, 12 validation frames and 62 other test frames. A cubic RGB renderer with regularization `1e-6` was selected using the mean validation scores across styles. Its validation RMSE was 0.00387, compared with 0.00908 for the best quadratic candidate. Test means for the selected renderer were 0.00321 (Sunlit), 0.00261 (Coastal) and 0.00462 (Soft film). These benchmark values are at maximum side 512 before JPEG export and before the variation/normalization stage. See `benchmark-final.json`.

## What the test establishes

Three looks were authored with independent Pillow tone curves and HSV color adjustments. The fitter saw each original/target image pair, not the target settings. It recovered one compact color mapping per lighting group. Deterministic scene-wide and individual exposure-like offsets, warm/cool shifts, small green tint and contrast differences were then applied to full-resolution copies (seed `20261030`). These are bounded global logit tone curves, not physical camera/RAW simulations.

The normalizer received only varied images and one intended clean reference per group. It used geometrically verified SIFT matches, aligned dense samples with a patch fallback, and robust graph fitting of global contrast and RGB offsets. Geometric transforms were used only to measure correspondences; exported geometry was preserved. Estimated corrections were replayed through the existing photo-correction mechanism before the scene style.

The evaluator separately read the withheld clean counterparts and injected settings. Its intended baseline is the same normalizer applied to the unperturbed album. This tests stability against injected variation; it does not independently establish ideal aesthetic normalization. Absolute scene baselines depend on the chosen references, and a shared style recipe supplies album intent. There is no independent absolute calibration inferred between disconnected scenes.

The variance family matches the corrector’s global model, making this a favorable controlled test. The scene splits, styles and algorithm were developed on this album. One shaded close-up has a separate reference and no within-scene transfer evidence. Human-edited counterparts and unseen albums were not evaluated in this showcase. Local relighting, clipping recovery and subject-mask accuracy remain outside its scope.

## Verification

- **Local tests:** 42 passed (`tests-final.log`), including exact cubic replay, normalization recovery/abstention, correction and replay with injected truth removed, and the complete showcase workflow on a synthetic fixture. Both workflow tests passed again after the original HTML fix (`workflow-final-tests.log`) and after the eight-stage walkthrough/local-example update (`walkthrough-tests.log`).
- **Real-album end to end:** 306 exported images verified, including 264 album outputs; zero file problems. Checks cover source hashes, coverage, complete decoding, original oriented dimensions, and ICC/EXIF presence (`verification.json`).
- **Dataset preservation:** all 88 originals and 88 supplied edits match their committed Git LFS SHA-256 identities (`dataset-integrity.json`).
- **Clipping review:** no sampled near-limit increase flags. The fitted mappings still clamp some pixels; the maximum sampled out-of-gamut fractions were 9.75% (Sunlit), 15.08% (Coastal), and 12.98% (Soft film), including bright regions already near limits. `worst-clamping-review.jpg` compares selected strongest cases with targets; the heuristic does not establish highlight/detail quality.
- **Showcase:** all 886 previews, 264 full-resolution album links and the separately copied historical person export passed validation; 1,332 local files were checked (`artifact-verification.json`). The expanded eight-stage walkthrough passed 43 browser checks at desktop and 390-pixel width, covering all 14 groups, three styles, saved artifacts, keyboard navigation, calibration/singleton disclosures, local-edit scope, presentation and gallery/inspector regressions (`walkthrough-browser-qa.json`). No console warnings/errors, broken loaded images or horizontal overflow were observed. The original 20-check report is retained as `browser-qa.json`. Browser QA does not cover every browser or device.
- **Portability:** the current showcase and attached person-experiment file references are repo-relative; website links remain relative to the showcase folder. A relocated copy passed all 1,334 HTTP path checks, preview/artifact verification, and independent checks of all 306 full-resolution exports (`portability-verification.json`). Four affected tests passed, including replay after relocation and rejection of changed inputs (`portability-tests.log`). These relocation checks ran on macOS; the documented Windows server command was not executed.
- **Package:** `package-verification.json` records the code ZIP checksum and byte comparison of every entry with current source.
- **CI:** not run. **Deployment:** not performed. This is an offline local deliverable.

## What the earlier agent achieved, and what remains

The earlier agent built the grouping proposals, deterministic target/prompt vocabulary, multiple reconstruction renderers, full-resolution replay, correction layers and integrity/review tooling. It also explored human-edit fitting and face/person-relative lighting. The resumed person experiment completed two illustrative groups with 13 exports; imperfect masks remain visible and its synthetic targets share those masks.

This continuation adds the cubic renderer, three independently authored style targets, reviewed 14-group plan, synthetic global settings differences, image-matched normalization, full-album outputs and an offline presentation. The showcase now follows a selected photo through Group → Style → Fit → Vary → Normalize → Replay → Album → Verify, showing actual inputs, estimates, saved mappings and exports. A separate local-light chapter displays an original, person mask, overlay and historical person-aware export. Face-region and person-mask edits exist in earlier workflows; neither is applied to the three 88-photo styled albums.

The next research work is aligned held-out evaluation of the 88 supplied human edits, additional albums with photographer review, reliable scene/reference selection, and local illumination or subject-aware correction. These controlled results support the showcase and executable pipeline; they do not establish professional quality or generalization.
