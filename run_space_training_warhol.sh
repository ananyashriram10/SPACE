#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY_ARTIST="Andy Warhol" \
TARGET_PROMPTS_PATH="data/andy_warhol_prompts.csv" \
TARGET_ARTIST_FILTER="Andy Warhol" \
ERASE_CONCEPT="Andy Warhol" \
EXP_NAME="Andy_Warhol" \
DEVICE="${DEVICE:-cuda:0}" \
SAVE_DIR="${SAVE_DIR:-space-models/sd}" \
STAGE1_STEPS="${STAGE1_STEPS:-300}" \
STAGE2_STEPS="${STAGE2_STEPS:-100}" \
bash "$ROOT/run_space_training.sh"
