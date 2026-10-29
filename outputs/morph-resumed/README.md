# Resumed person-aware scene experiment

This continues the last unfinished task in the main session log in `27/`: multi-scene person-aware fit, replay and review. The sample has six lawn photos and seven building photos, with one fitted representative per scene. These are illustrative group assignments inherited from the earlier experiment.

## Results

- `review/index.html`: original/edited comparisons grouped by scene, with links to all 13 full-resolution exports.
- `mask-review/index.html`: each original, soft person mask and overlay.
- `full/`: exports at original oriented dimensions, with ICC and EXIF present.
- `verification.json`: coverage against the local groups and fits, complete image decoding, dimensions, metadata presence and 30 local review links checked.
- `transfer-evaluation.json`: 11 held-out photos excluded from fitting. Mean RGB RMSE fell from 0.0380 for the source to 0.00563 for the recovered mapping; all 11 improved against the constructed per-photo targets. Evaluation is at fit resolution before JPEG export, using masks shared by target and reconstruction. It does not validate the masks, artistic quality, human edits or unseen-album performance.

Visual inspection of the thumbnail mask sheets still shows partial coverage of some distant people. No nearly-empty-mask flags fired, which illustrates why that flag is insufficient to establish mask quality. The edited look is subtle. Automatic normalization remains unresolved; no manual corrections were supplied to this run.

## Recovery

The 13 originals were verified against source hashes in the archived `person-two-scene-masks.json`. Archived masks and recent exports were absent from their folders in this checkout, so they were regenerated with the existing local Swift Vision helper. The two difficult lawn photos (`DSC_0947-13.jpg` and `DSC_0948-14.jpg`) use regenerated multiscale masks; the remaining photos use whole-frame masks. This reproduces the previous replacement choices, not a claim of byte-identical masks across macOS versions. The run refits using current inputs and preserves checksum validation.

## Repeat fitting and rendering

From the repository root, with the local virtual environment installed, use new output directory names:

```bash
.venv/bin/python outputs/morph-starter/person_album.py fit \
  outputs/morph-resumed/groups.json outputs/morph-resumed/masks.json \
  --prompt 'warm and slightly soft contrast and lift shadows and brighten people' \
  --out outputs/morph-resumed/fits-next
.venv/bin/python outputs/morph-starter/person_album.py apply \
  outputs/morph-resumed/fits-next --out outputs/morph-resumed/full-next
.venv/bin/python outputs/morph-starter/review_person_album.py \
  outputs/morph-resumed/full-next --out outputs/morph-resumed/review-next
```

Add `--preview-samples 2` to `apply` to review a small full-resolution sample per scene first. Add `--corrections corrections.json` to apply reviewed photo, scene and album corrections. The saved manifests use paths relative to the repository root. Run these commands from that root and keep the dataset/output layout intact when cloning or moving it. Source and mask hashes still validate the inputs.

## Verification performed

The existing 38-test starter suite passed before changes (`tests.log`). After the combined review was added, all three affected person-scene/album tests passed (`person-tests.log`), including preview/full review links and rejecting inconsistent counts or corrupted exports without leaving a partial review. The real local run rendered and verified all 13 photos. No CI or deployment was run. The remaining 75 originals were not rerendered in this continuation; earlier full-album results remain historical evidence. Supplied human edits and unseen albums were not reevaluated here.

## Next research steps

1. Inspect masks and full-resolution subject edges, especially distant people, before treating this subject-aware path as reliable.
2. Evaluate scene grouping and representative choice against manually reviewed assignments and additional albums.
3. Evaluate aligned human-edited pairs on held-out photos. Separate alignment issues and individual artistic intent from reconstruction error.
4. Develop and validate reliable per-photo normalization; existing metadata/face heuristics had mixed outcomes.
5. Curate the strongest reconstruction and scene-transfer examples into the research showcase, with failures and scope made clear.
