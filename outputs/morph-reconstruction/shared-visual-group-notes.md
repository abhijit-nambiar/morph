# One shared edit per visually proposed group

This test compares a single polynomial mapping per visual subgroup with a single mapping per chronological scene. Each subgroup's reference is the eligible original/edit pair closest to its visual medoid; the baseline uses the previously chosen chronological reference. Reference choices use the originals and group assignments, not the held-out edits. Each tested photo is excluded if it served as either reference.

Among 23 jointly held-out photos in 10 visual groups and seven chronological scenes, mean RGB RMSE was 0.0618 with the visual group reference and 0.0685 with the chronological reference. The visual version had lower error for 12 photos, higher error for seven, and equal error for four. This is a restricted subset of the 46 same-size pairs: references and groups without another eligible pair cannot contribute held-out observations. The results cannot be assumed to cover the rest of the album.

The comparison sheet shows four held-out photos from visual group 15. Both reconstructed columns use one mapping across the four photos. The supplied edits are shown as a reference. The sample illustrates a limitation of this global color model: it does not recover all appearance changes in the supplied edits.

This is an exploratory, reduced-resolution pixel comparison on one album. Groups and original/edit alignment are unverified; the threshold was informed by this album, and edits can include individual intent. RGB error does not judge professional or aesthetic quality. The visual group reference is chosen from available edited examples; the prompt workflow would need to generate a target for the chosen original. Normalization and additional-album validation remain open.
