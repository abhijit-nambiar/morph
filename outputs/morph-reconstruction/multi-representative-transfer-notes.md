# Multiple examples for a scene: exploratory test

This experiment fits a polynomial color mapping separately to each available original/edit pair, then averages the mappings from other photos in the same proposed scene and applies that averaged mapping to the held-out original. The held-out original/edit pair never participates in the fit. Unlike the one-example pipeline, this experiment needs multiple edited examples in each scene and does not provide a per-photo normalization step.

In eight proposed scenes, 46 photos were each held out once. The mean held-out RGB RMSE was 0.0595 for the averaged mapping and 0.0672 when averaging the errors of the alternative single-example mappings. This latter comparison benefits pooling by construction: averaging related predictions tends to reduce average squared error, and the alternatives are not a specific representative selection policy.

For a more concrete comparison, retain each scene's pre-existing representative and evaluate the remaining 38 photos. The averaged mapping achieved mean error 0.0606 versus 0.0664 for the fixed representative, and had lower error on 22 of 38 photos. The two methods use different amounts of edited training data, so this is evidence about a possible multi-example workflow, not an equivalent one-prompt replacement or proof of better results for every photo.

The evaluation is limited to one album and 128-pixel images. The scene groups and alignment are unverified; the supplied edits may have different intentions, and pixel error cannot assess artistic quality. A useful next step is a small fixed-number-of-examples study with scenes held out across albums, alongside full-resolution human review, before incorporating this into the main workflow.
