#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY_ARTIST="Andy Warhol" \
EXP_NAME="Andy_Warhol" \
TARGET_PROMPTS_PATH="data/andy_warhol_prompts.csv" \
TARGET_ARTIST_FILTER="Andy Warhol" \
NUM_SAMPLES="${NUM_SAMPLES:-10}" \
DEVICE="${DEVICE:-cuda:0}" \
bash "$ROOT/run_space_images.sh"
