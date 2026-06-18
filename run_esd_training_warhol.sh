#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
DEVICE="${DEVICE:-cuda:0}"
SAVE_DIR="${SAVE_DIR:-esd-models/sd}"
ITERATIONS="${ITERATIONS:-200}"
LR="${LR:-5e-5}"
NEGATIVE_GUIDANCE="${NEGATIVE_GUIDANCE:-2}"
mkdir -p "$SAVE_DIR" "$ROOT/logs"
"$PYTHON_BIN" "$ROOT/esd_sd.py" \
  --erase_concept "Andy Warhol" \
  --train_method "esd-x" \
  --iterations "$ITERATIONS" \
  --lr "$LR" \
  --negative_guidance "$NEGATIVE_GUIDANCE" \
  --save_path "$SAVE_DIR" \
  --device "$DEVICE" \
  2>&1 | tee "$ROOT/logs/esd_train_andy_warhol.log"
