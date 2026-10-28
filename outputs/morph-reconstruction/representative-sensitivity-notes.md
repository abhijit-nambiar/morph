# Representative sensitivity: research note

A shared edit estimated from a single photo can absorb that photo's particular exposure or its individual treatment. We tested the effect by fitting a polynomial color mapping to each eligible original/edit pair in turn, then applying it to the other eligible photos in that proposed scene. A held-out photo never served as its own fit. The study used eight proposed scenes, 46 same-dimension pairs, 128-pixel fitting/evaluation images, and regularization 0.001.

For each held-out photo, the range of RGB RMSE across alternative representatives averaged 0.0558 on a 0–1 scale (median 0.0468). For DSC_1033-30.jpg in scene-03, the range was 0.0303–0.1674, compared with 0.0764 for the unedited original. The comparison sheet shows scene-03's original and supplied edit pairs; the originals vary in exposure and composition, and the supplied edits may include individual adjustments. These observations are consistent with a representative's individual correction being mixed into the recovered scene mapping, but do not establish the cause.

The groups have not been validated, matching dimensions do not establish pixel alignment, and the supplied edits may reflect different intentions. Pixel RMSE is not a judgment of aesthetic quality. The minimum and maximum are retrospective diagnostics, not a method for picking a representative without access to other edited examples.

A promising next experiment is to separate a shared scene look from per-photo normalization, and evaluate it on held-out photos and additional albums. This evidence does not yet justify enabling automatic normalization.
