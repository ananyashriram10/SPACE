#!/bin/bash
# Generate SPACE-v2 images for the Van Gogh benchmark.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/use_venv.sh"

ONLY_ARTIST="${ONLY_ARTIST:-Van Gogh}"
DEVICE="${DEVICE:-cuda:0}"
WEIGHTS_DIR="${WEIGHTS_DIR:-space-models/sd-v2}"
OUT_DIR="${OUT_DIR:-results/space_v2}"
EXP_NAME="${EXP_NAME:-Van_Gogh}"
TARGET_PROMPTS_PATH="${TARGET_PROMPTS_PATH:-data/vangogh_prompts.csv}"
TARGET_ARTIST_FILTER="${TARGET_ARTIST_FILTER:-Vincent van Gogh}"

if [ "$ONLY_ARTIST" != "Van Gogh" ]; then
  echo "SPACE-v2 image generation is currently scoped to Van Gogh only. Received ONLY_ARTIST=$ONLY_ARTIST"
  exit 1
fi

CKPT="$ROOT/$WEIGHTS_DIR/space-$EXP_NAME.safetensors"
if [ ! -f "$CKPT" ]; then
  echo "Missing SPACE-v2 checkpoint: $CKPT"
  echo "Run: python space_v2/space_sd_v2.py --erase_concept 'Van Gogh' --exp_name Van_Gogh --target_prompts_path data/vangogh_prompts.csv --target_artist_filter 'Vincent van Gogh'"
  exit 1
fi

echo "============================================================"
echo "  SPACE-v2 Image Generation"
echo "============================================================"
echo "Checkpoint: $CKPT"
echo "Output root: $ROOT/$OUT_DIR"
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
  --num_samples "${NUM_SAMPLES:-5}" \
  --artist_filter "$TARGET_ARTIST_FILTER" \
  2>&1 | tee "$ROOT/logs/space_v2_images_${EXP_NAME,,}.log"

mkdir -p "$ROOT/results/provenance/space_v2"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
payload = {
    "method": "SPACE-v2",
    "artist": "Van Gogh",
    "run_type": "research-method",
    "checkpoint": "space-models/sd-v2/space-Van_Gogh.safetensors",
    "outputs": ["results/space_v2/space-Van_Gogh"],
    "prompts": "data/vangogh_prompts.csv",
    "improvements": ["B: attn1+FF LoRA targets", "C: dense 0.3-0.6 trajectory", "J: per-prompt loss weighting"],
}
out = root / "results/provenance/space_v2/space_v2-Van_Gogh.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote provenance: {out}")
PY

echo ""
echo "SPACE-v2 image generation complete."
echo "Images: $OUT_DIR/space-$EXP_NAME"
