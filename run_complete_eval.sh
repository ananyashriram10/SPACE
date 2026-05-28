#!/bin/bash
# ============================================================
# run_complete_eval.sh
# Brings ALL methods to NUM_SAMPLES per prompt and runs eval.
# Also generates CA FID on COCO 2K.
#
# Usage:
#   bash run_complete_eval.sh            # default 10 samples
#   NUM_SAMPLES=20 bash run_complete_eval.sh
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"

NUM_SAMPLES="${NUM_SAMPLES:-10}"
DEVICE="${DEVICE:-cuda:0}"

ESD_PT="$ROOT/esd-weights/art/diffusers-VanGogh-ESDx1-UNET.pt"
UCE_CKPT="$ROOT/baseline-models/uce/uce-Van_Gogh.safetensors"
CA_DIR="$ROOT/baselines/external/concept-ablation/compvis"
CA_CKPT="$ROOT/baseline-assets/concept_ablation/pretrained_models/sd-v1-4.ckpt"
CA_DELTA="$ROOT/baseline-models/concept_ablation_compvis/official_weights/concept_ablation-Van_Gogh.ckpt"
CA_PROMPTS="$ROOT/baselines/cache/official_prompts/concept_ablation_compvis/concept_ablation-Van_Gogh.txt"
CA_OUTDIR="$ROOT/results/concept_ablation_compvis/concept_ablation-Van_Gogh"
VANGOGH_CSV="$ROOT/data/vangogh_prompts.csv"
COCO_CSV="$ROOT/data/coco_30k.csv"
COCO_N=2000

echo "============================================================"
echo "  Complete evaluation — ${NUM_SAMPLES} samples/prompt"
echo "  Device: $DEVICE"
echo "============================================================"
echo ""

# ── 1. Baseline ──────────────────────────────────────────────
echo "[1/7] Baseline: generating up to ${NUM_SAMPLES} samples/prompt..."
"$PYTHON_BIN" "$ROOT/generate_esd_old.py" \
  --prompts_path "$VANGOGH_CSV" \
  --save_path "$ROOT/results/baseline/vangogh" \
  --num_samples "$NUM_SAMPLES"

# ── 2. ESD-x ─────────────────────────────────────────────────
echo "[2/7] ESD-x: generating up to ${NUM_SAMPLES} samples/prompt..."
if [ ! -f "$ESD_PT" ]; then
  echo "  WARNING: ESD-x weights missing at $ESD_PT — skipping"
else
  "$PYTHON_BIN" "$ROOT/generate_esd_old.py" \
    --esd_path "$ESD_PT" \
    --prompts_path "$VANGOGH_CSV" \
    --save_path "$ROOT/results/erased/vangogh" \
    --num_samples "$NUM_SAMPLES"
fi

# ── 3. SPACE ─────────────────────────────────────────────────
echo "[3/7] SPACE: generating up to ${NUM_SAMPLES} samples/prompt..."
NUM_SAMPLES="$NUM_SAMPLES" DEVICE="$DEVICE" bash "$ROOT/run_space_images.sh"

# ── 4. UCE ───────────────────────────────────────────────────
echo "[4/7] UCE: generating up to ${NUM_SAMPLES} samples/prompt..."
"$PYTHON_BIN" "$ROOT/evalscripts/generate-images.py" \
  --base_model CompVis/stable-diffusion-v1-4 \
  --esd_path "$UCE_CKPT" \
  --prompts_path "$VANGOGH_CSV" \
  --save_path "$ROOT/results/uce" \
  --num_samples "$NUM_SAMPLES" \
  --num_inference_steps 20 \
  --guidance_scale 7.5 \
  --device "$DEVICE"

# ── 5. CA: clear + regenerate with n_copies ──────────────────
echo "[5/7] Concept Ablation: regenerating with n_copies=${NUM_SAMPLES}..."
if [ ! -f "$CA_DIR/sample.py" ]; then
  echo "  WARNING: CA repo not found at $CA_DIR — skipping"
elif [ ! -f "$CA_DELTA" ]; then
  echo "  WARNING: CA delta not found at $CA_DELTA — skipping"
else
  rm -f "$CA_OUTDIR/samples/"*.png
  (cd "$CA_DIR" && "$PYTHON_BIN" sample.py \
    --ckpt "$CA_CKPT" \
    --delta_ckpt "$CA_DELTA" \
    --from-file "$CA_PROMPTS" \
    --ddim_steps 100 --scale 6.0 --ddim_eta 1.0 \
    --outdir "$CA_OUTDIR" \
    --n_samples 1 --n_copies "$NUM_SAMPLES" \
    --skip_grid --metadata --seed 42)
fi

# ── 6. CA FID: 2K COCO images ────────────────────────────────
# NOTE: CA uses CompVis LDM (100 DDIM steps) vs diffusers 20-step
# baseline. FID is directionally useful but not strictly comparable
# to SPACE/UCE/ESD-x FID. Flag this caveat when reporting.
echo "[6/7] CA FID: generating ${COCO_N} COCO images for FID..."
if [ ! -f "$CA_DIR/sample.py" ] || [ ! -f "$CA_DELTA" ]; then
  echo "  WARNING: CA not available — skipping CA FID"
else
  COCO_TXT="/tmp/coco_${COCO_N}_prompts.txt"
  "$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv("$COCO_CSV").head($COCO_N)
with open("$COCO_TXT", "w") as f:
    f.write("\n".join(df.prompt.tolist()))
print(f"Wrote {len(df)} COCO prompts to $COCO_TXT")
PY

  CA_FID_TMPDIR="$ROOT/results/fid/ca_tmp"
  CA_FID_DIR="$ROOT/results/fid/concept_ablation"
  rm -rf "$CA_FID_TMPDIR"
  mkdir -p "$CA_FID_DIR"

  (cd "$CA_DIR" && "$PYTHON_BIN" sample.py \
    --ckpt "$CA_CKPT" \
    --delta_ckpt "$CA_DELTA" \
    --from-file "$COCO_TXT" \
    --ddim_steps 100 --scale 6.0 --ddim_eta 1.0 \
    --outdir "$CA_FID_TMPDIR" \
    --n_samples 1 --n_copies 1 \
    --skip_grid --seed 42)

  # sample.py saves to {outdir}/samples/ — move up to FID dir
  mv "$CA_FID_TMPDIR/samples/"*.png "$CA_FID_DIR/"
  rm -rf "$CA_FID_TMPDIR"
  echo "  CA FID images ready: $(ls $CA_FID_DIR | wc -l) images"
fi

# ── 7. Evaluate ──────────────────────────────────────────────
echo "[7/7] Running evaluation..."
rm -f "$ROOT/results/evaluation/metrics_cache_v5_van_gogh.json"
"$PYTHON_BIN" "$ROOT/evaluate.py" --only-artist "Van Gogh"

echo ""
echo "============================================================"
echo "  Done. Results in results/evaluation/"
echo "============================================================"
