# Morph research preview: presenter notes

## 1. Scene edit reconstruction

The research question is whether an edit applied to one representative photo can be recovered as a repeatable transformation and transferred to other photos in the same scene. The current work focuses on lighting and color. The presenter can show the source image, a target look, the recovered edit, and several related photos that receive that same recovered mapping. The results are a research preview rather than a finished editing product.

## 2. Pipeline overview

First, the system proposes groups of photos and identifies a representative for each group. A prompt produces an edited target for that representative. The reconstruction step compares the representative and target to fit an executable color-and-lighting mapping. The system applies the saved mapping to other photos in the same group. Finally, a reviewer can adjust an individual photo, a scene, or the whole album and inspect the results. These stages are separate: transferring the same edit alone does not ensure every photo has the same exposure.

## 3. Grouping photos into scenes

The current grouping baseline orders photos by capture time and starts a new group after a configurable time gap or camera change. It chooses a representative using basic image statistics. This is a proposal, not semantic scene recognition: a time-based group can contain different settings or lighting. In the sample, the original second group mixed lawn photos with photos near a building. We created an illustrative manual split and kept every photo. Group membership and representative selection can be edited before fitting.

## 4. Prompt and target edit

For the example, a rule-based interpreter maps the phrase ‘warm, slightly soft contrast and lift shadows’ to a small set of target settings. An independent deterministic image-editing routine applies tonal and color operations to the representative to create the target on the right. The prompt vocabulary is explicit and limited. This target is a controlled test of reconstruction, not a photograph edited by a human. The same workflow can also fit selected existing edited representatives when comparable files are supplied.

## 5. Recovering the edit

The original and target form the pair used for fitting. The optimizer adjusts a separate renderer’s color and lighting parameters to reduce the difference between its output and the target. The demo uses a bounded color model and a damped least-squares optimization method. Fitting happens on a reduced image with a maximum side of 320 pixels; the saved mapping is then applied to the full-resolution original. The mapping is a reproducible approximation of the visual change. It does not prove recovery of the target editor’s original settings.

## 6. Transfer within the scene

The person presenting can point out that this is a different photo from the one used to fit the mapping. The saved scene mapping is applied without refitting to this image. A controlled check separately generates the same synthetic target look for the other photo, then compares the output with that target. The resulting pixel error is useful for checking reconstruction in this experiment, but it is not a score for taste or professional quality.

## 7. Scene set: three group photos

The top row contains three original photos from the same proposed scene. The bottom row shows each corresponding photo after the same recovered scene mapping. The middle photo was the representative in the full-album run; the other two received the mapping as transfers. This set comes from the experimental face-aware album path, which also uses face boxes detected separately in each image. It is a different run from the global color example on the previous slide, although the same principle of a shared scene mapping applies.

## 8. Scene set: two portraits

Each pair shows an original portrait and its result after the scene mapping. Both photos belong to the same proposed scene and share the mapping fitted from another representative. In the face-aware variant, a smooth lighting region is placed using each photo’s detected face boxes, allowing that part of the effect to follow detected faces as their positions change. The boxes are not person segmentation: the mask also affects nearby surroundings, and detection accuracy needs visual review.

## 9. Normalization and review

The left image shows a recovered scene look. The right adds illustrative corrections at three levels: a photo exposure change, a scene shadow adjustment, and a small album-wide temperature change. The order is photo correction, fitted scene mapping, scene correction, and album correction. The system can propose exposure candidates based on camera metadata, but these require review. In previous tests, proposed brightness changes sometimes improved the match and sometimes made it worse, so the workflow does not claim reliable automatic normalization.

## 10. What the sample album shows

The prototype rendered all 88 sample JPEGs at their original oriented dimensions using nine proposed groups. For the synthetic face-aware test, one representative per group was used for fitting and the remaining 79 photos were checked against independently recomputed per-photo synthetic targets. All 79 had lower RGB error than the unedited source; mean error decreased from approximately 0.0295 to 0.0061 on a 0–1 scale. The comparison was made at fitting resolution before JPEG export. Target and model use the same detected face boxes, the grouping is unverified, and these measurements do not assess artistic quality.

## 11. Research status and next steps

The demonstrated pieces work together on the sample album, with editable groups, deterministic target generation, mapping recovery, scene transfer, and manual corrections. The main open questions are whether scene boundaries and representative choices remain reliable on other albums, whether per-photo normalization can be proposed safely, and how people judge the outputs. Next, evaluate on unseen albums and aligned human-edited examples, inspect face detections and full-resolution images, and collect photographer feedback. The current prompt vocabulary and local masks are deliberately limited.

