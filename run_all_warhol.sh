#!/bin/bash
# Master pipeline: end-to-end Andy Warhol evaluation across all methods.
# Steps: baseline → SPACE train → ESD-x train → UCE train → CA train →
#        SPACE images → ESD-x images → UCE images → CA images → SPACE FID
# Each step is skippable via SKIP_<STEP>=1 env vars.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  Andy Warhol — Full Evaluation Pipeline"
echo "============================================================"

# ── Step 1: Baseline ────────────────────────────────────────────
if [ "${SKIP_BASELINE:-0}" != "1" ]; then
  echo ""
  echo "[1/8] Generating baseline images (SD v1.4)..."
  cd "$ROOT"
  python generate_esd_old.py \
    --prompts_path data/andy_warhol_prompts.csv \
    --save_path results/baseline/andy_warhol \
    --num_samples 10 \
    2>&1 | tee logs/baseline_andy_warhol.log
  echo "[1/8] Baseline DONE"
else
  echo "[1/8] SKIPPED (SKIP_BASELINE=1)"
fi

# ── Step 2: SPACE training ──────────────────────────────────────
if [ "${SKIP_SPACE_TRAIN:-0}" != "1" ]; then
  echo ""
  echo "[2/8] Training SPACE..."
  bash "$ROOT/run_space_training_warhol.sh"
  echo "[2/8] SPACE training DONE"
else
  echo "[2/8] SKIPPED (SKIP_SPACE_TRAIN=1)"
fi

# ── Step 3: ESD-x training ──────────────────────────────────────
if [ "${SKIP_ESD_TRAIN:-0}" != "1" ]; then
  echo ""
  echo "[3/8] Training ESD-x..."
  bash "$ROOT/run_esd_training_warhol.sh"
  echo "[3/8] ESD-x training DONE"
else
  echo "[3/8] SKIPPED (SKIP_ESD_TRAIN=1)"
fi

# ── Step 4: UCE training ─────────────────────────────────────────
if [ "${SKIP_UCE_TRAIN:-0}" != "1" ]; then
  echo ""
  echo "[4/8] Training UCE..."
  ONLY_ARTIST="Andy Warhol" bash "$ROOT/baselines/run_uce_training.sh"
  echo "[4/8] UCE training DONE"
else
  echo "[4/8] SKIPPED (SKIP_UCE_TRAIN=1)"
fi

# ── Step 5: CA training ──────────────────────────────────────────
if [ "${SKIP_CA_TRAIN:-0}" != "1" ]; then
  echo ""
  echo "[5/8] Training Concept Ablation (diffusers)..."
  ONLY_ARTIST="Andy Warhol" CA_SAMPLES=10 bash "$ROOT/baselines/run_concept_ablation_diffusers_training.sh"
  echo "[5/8] CA training DONE"
else
  echo "[5/8] SKIPPED (SKIP_CA_TRAIN=1)"
fi

# ── Step 6: SPACE images ─────────────────────────────────────────
if [ "${SKIP_SPACE_IMAGES:-0}" != "1" ]; then
  echo ""
  echo "[6/8] Generating SPACE images..."
  bash "$ROOT/run_space_images_warhol.sh"
  echo "[6/8] SPACE images DONE"
else
  echo "[6/8] SKIPPED (SKIP_SPACE_IMAGES=1)"
fi

# ── Step 7: ESD-x images ─────────────────────────────────────────
if [ "${SKIP_ESD_IMAGES:-0}" != "1" ]; then
  echo ""
  echo "[7/8] Generating ESD-x images..."
  bash "$ROOT/run_esd_images_warhol.sh"
  echo "[7/8] ESD-x images DONE"
else
  echo "[7/8] SKIPPED (SKIP_ESD_IMAGES=1)"
fi

# ── Step 8: UCE images ───────────────────────────────────────────
if [ "${SKIP_UCE_IMAGES:-0}" != "1" ]; then
  echo ""
  echo "[8/8] Generating UCE images..."
  ONLY_ARTIST="Andy Warhol" UCE_SAMPLES=10 bash "$ROOT/baselines/run_uce_images.sh"
  echo "[8/8] UCE images DONE"
else
  echo "[8/8] SKIPPED (SKIP_UCE_IMAGES=1)"
fi

# ── Step 9: CA images ────────────────────────────────────────────
if [ "${SKIP_CA_IMAGES:-0}" != "1" ]; then
  echo ""
  echo "[9/9] Generating CA images..."
  ONLY_ARTIST="Andy Warhol" CA_SAMPLES=10 bash "$ROOT/baselines/run_concept_ablation_diffusers_images.sh"
  echo "[9/9] CA images DONE"
else
  echo "[9/9] SKIPPED (SKIP_CA_IMAGES=1)"
fi

# ── Step 10: SPACE FID (30k COCO) ───────────────────────────────
# Note: ESD-x/UCE/CA FID values are taken from their respective papers.
if [ "${SKIP_FID:-0}" != "1" ]; then
  echo ""
  echo "[FID] Generating 30k FID samples for SPACE..."
  bash "$ROOT/run_fid_warhol_space.sh"
  echo "[FID] SPACE FID samples DONE"
else
  echo "[FID] SKIPPED (SKIP_FID=1)"
fi

echo ""
echo "============================================================"
echo "  Andy Warhol pipeline COMPLETE"
echo "  Run evaluate.py --only-artist 'Andy Warhol' to score."
echo "============================================================"
