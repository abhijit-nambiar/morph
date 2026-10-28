# One versus two prompt-generated references

For each proposed scene, we generated targets from the same prompt for the existing representative and the first other photo in group order. We fitted a polynomial mapping to each target and averaged the two mappings. We compared this with the first representative's mapping on 70 other photos; neither reference appears in the evaluated set. This was repeated for three global color-and-tone prompts at a maximum image side of 128 pixels.

| Prompt | One reference mean RMSE | Two references mean RMSE | Photos improved / worsened |
| --- | ---: | ---: | ---: |
| Warm, softer contrast, lifted shadows | 0.00607 | 0.00620 | 34 / 36 |
| Cool, strong contrast, deeper shadows | 0.00715 | 0.00682 | 47 / 23 |
| Warmer, reduced highlights, muted blues | 0.00884 | 0.00842 | 44 / 26 |

The extra reference helped on average in two prompts and slightly hurt on the warm prompt. This is not consistent enough to choose it as the default. The group assignment and second-reference selection have not been validated. All targets use the same deterministic generator, so this tests approximation of that generator rather than aesthetic quality or real-edit transfer. It does not test local edits or normalization, and results need full-resolution and additional-album evaluation.
