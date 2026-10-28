#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! python3 -c 'import numpy, PIL' >/dev/null 2>&1; then
    printf '%s\n' 'Python 3, NumPy and Pillow are required. See README.md.' >&2
    exit 1
fi
output_dir="${1:-demo-output}"
OPENBLAS_NUM_THREADS=1 python3 code/morph.py album groups.json \
  --target-engine pillow-soft \
  --prompt 'warm, slightly soft contrast and lift shadows' \
  --renderer gamut --solver gauss-newton --size 320 \
  --preview-only --preview-samples 3 --full-res-reference \
  --evaluate-transfer --out "$output_dir"
printf '\nResults: %s/scene-01\n' "$output_dir"
