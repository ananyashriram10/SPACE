#!/bin/bash
# Generate SPACE-v3 images for the Van Gogh benchmark.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/use_venv.sh"

ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS_DIR="${WEIGHTS_DIR:-space-models/sd-v3}"
OUT_DIR="${OUT_DIR:-results/space_v3}"
EXP_NAME="${EXP_NAME:-Van_Gogh_v3}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-space-Van_Gogh}"
TARGET_PROMPTS_PATH="${TARGET_PROMPTS_PATH:-data/vangogh_prompts.csv}"
TARGET_ARTIST_FILTER="${TARGET_ARTIST_FILTER:-Vincent van Gogh}"

if [ "$ONLY_ARTIST" != "Van Gogh" ]; then
  echo "SPACE-v3 image generation is currently scoped to Van Gogh only. Received ONLY_ARTIST=$ONLY_ARTIST"
  exit 1
fi

CKPT="$ROOT/$WEIGHTS_DIR/space-$EXP_NAME.safetensors"
if [ ! -f "$CKPT" ]; then
  echo "Missing SPACE-v3 checkpoint: $CKPT"
  echo "Run: bash space_v3/run_space_v3_vangogh.sh"
  exit 1
fi

echo "============================================================"
echo "  SPACE-v3 Image Generation"
echo "============================================================"
echo "Checkpoint: $CKPT"
echo "Output root: $ROOT/$OUT_DIR"
echo "Output folder: $OUTPUT_MODEL_NAME"
echo ""

mkdir -p "$ROOT/logs"

"$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
  --base_model CompVis/stable-diffusion-v1-4 \
  --esd_path "$CKPT" \
  --prompts_path "$ROOT/$TARGET_PROMPTS_PATH" \
  --save_path "$ROOT/$OUT_DIR" \
  --device "$DEVICE" \
  --guidance_scale "${EVAL_GUIDANCE_SCALE:-7.5}" \
  --num_inference_steps "${EVAL_STEPS:-20}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --artist_filter "$TARGET_ARTIST_FILTER" \
  --model_name_override "$OUTPUT_MODEL_NAME" \
  2>&1 | tee "$ROOT/logs/space_v3_images_${EXP_NAME,,}.log"

mkdir -p "$ROOT/results/provenance/space_v3"
EXP_NAME="$EXP_NAME" OUT_DIR="$OUT_DIR" OUTPUT_MODEL_NAME="$OUTPUT_MODEL_NAME" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path.cwd()
exp_name = os.environ["EXP_NAME"]
out_dir = os.environ["OUT_DIR"]
model_name = os.environ["OUTPUT_MODEL_NAME"]
payload = {
    "method": "SPACE-v3",
    "artist": "Van Gogh",
    "run_type": "research-method",
    "checkpoint": f"space-models/sd-v3/space-{exp_name}.safetensors",
    "outputs": [f"{out_dir}/{model_name}"],
    "prompts": "data/vangogh_prompts.csv",
}
out = root / "results" / "provenance" / "space_v3" / f"images_space-{exp_name}.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote provenance: {out}")
PY

echo ""
echo "SPACE-v3 image generation complete."
echo "Images: $OUT_DIR/$OUTPUT_MODEL_NAME"
