#!/bin/bash
# Generate SPACE-v1 images for the current artist benchmark.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"
ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS_DIR="${WEIGHTS_DIR:-space-models/sd}"
OUT_DIR="${OUT_DIR:-results/space}"
EXP_NAME="${EXP_NAME:-Van_Gogh}"
TARGET_PROMPTS_PATH="${TARGET_PROMPTS_PATH:-data/vangogh_prompts.csv}"
TARGET_ARTIST_FILTER="${TARGET_ARTIST_FILTER:-Vincent van Gogh}"

CKPT="$ROOT/$WEIGHTS_DIR/space-$EXP_NAME.safetensors"
if [ ! -f "$CKPT" ]; then
  echo "Missing SPACE checkpoint: $CKPT"
  echo "Run: bash run_space_training.sh"
  exit 1
fi

echo "============================================================"
echo "  SPACE-v1 Image Generation"
echo "============================================================"
echo "Checkpoint: $CKPT"
echo "Output root: $ROOT/$OUT_DIR"
echo ""

"$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
  --base_model CompVis/stable-diffusion-v1-4 \
  --esd_path "$CKPT" \
  --prompts_path "$ROOT/$TARGET_PROMPTS_PATH" \
  --save_path "$ROOT/$OUT_DIR" \
  --device "$DEVICE" \
  --guidance_scale "${EVAL_GUIDANCE_SCALE:-7.5}" \
  --num_inference_steps "${EVAL_STEPS:-20}" \
  --num_samples "${NUM_SAMPLES:-5}" \
  --artist_filter "$TARGET_ARTIST_FILTER" \
  2>&1 | tee "$ROOT/logs/space_images_${EXP_NAME,,}.log"

mkdir -p "$ROOT/results/provenance/space"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
payload = {
    "method": "SPACE",
    "artist": "Van Gogh",
    "run_type": "research-method",
    "checkpoint": "space-models/sd/space-Van_Gogh.safetensors",
    "outputs": ["results/space/space-Van_Gogh"],
    "prompts": "data/vangogh_prompts.csv",
}
out = root / "results/provenance/space/space-Van_Gogh.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote provenance: {out}")
PY

echo ""
echo "SPACE image generation complete."
echo "Images: $OUT_DIR/space-$EXP_NAME"
