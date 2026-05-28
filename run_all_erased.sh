#!/bin/bash
# ============================================================
# run_all_erased.sh — Generate erased images for all 6 artists
# using the paper's pre-trained ESD-x UNet checkpoints.
#
# All 6 jobs run in parallel.
# A40 (48GB): 6 × SD v1.4 fp16 ≈ 21GB — comfortable.
# If OOM, comment out 3 of the 6 jobs and run in two batches.
# ============================================================

set -e
WEIGHTS="esd-weights/art"
mkdir -p results/erased logs

# Verify all weight files exist before starting
REQUIRED=(
  "$WEIGHTS/diffusers-KellyMcKernan-ESDx1-UNET.pt"
  "$WEIGHTS/diffusers-VanGogh-ESDx1-UNET.pt"
  "$WEIGHTS/diffusers-TylerEdlin-ESDx1-UNET.pt"
  "$WEIGHTS/diffusers-KilianEng-ESDx1-UNET.pt"
  "$WEIGHTS/diffusers-ThomasKinkade-ESDx1-UNET.pt"
  "$WEIGHTS/diffusers-AjinDemiHuman-ESDx1-UNET.pt"
)
echo "Checking weight files..."
MIN_SIZE=3000000000  # 3 GB — a complete file is ~3.2 GB
for f in "${REQUIRED[@]}"; do
  if [ ! -f "$f" ]; then
    echo "  ERROR: Missing $f — run download_weights.sh first."
    exit 1
  fi
  SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
  if [ "$SIZE" -lt "$MIN_SIZE" ]; then
    echo "  ERROR: $f looks incomplete (${SIZE} bytes < 3 GB)."
    echo "         Re-run download_weights.sh to finish the download."
    exit 1
  fi
  echo "  OK: $f ($(du -sh "$f" | cut -f1))"
done

echo ""
echo "============================================================"
echo "  ESD Replication: Erased Image Generation (6 artists)"
echo "  Launching 6 parallel jobs..."
echo "============================================================"

python generate_esd_old.py \
  --esd_path "$WEIGHTS/diffusers-KellyMcKernan-ESDx1-UNET.pt" \
  --prompts_path data/kelly_prompts.csv \
  --save_path results/erased/kelly \
  2>&1 | tee logs/erased_kelly.log &
PID1=$!
echo "  [1/6] Kelly McKernan  — PID $PID1  (logs/erased_kelly.log)"

python generate_esd_old.py \
  --esd_path "$WEIGHTS/diffusers-VanGogh-ESDx1-UNET.pt" \
  --prompts_path data/vangogh_prompts.csv \
  --save_path results/erased/vangogh \
  2>&1 | tee logs/erased_vangogh.log &
PID2=$!
echo "  [2/6] Van Gogh        — PID $PID2  (logs/erased_vangogh.log)"

python generate_esd_old.py \
  --esd_path "$WEIGHTS/diffusers-TylerEdlin-ESDx1-UNET.pt" \
  --prompts_path data/short_niche_art_prompts.csv \
  --artist "Tyler Edlin" \
  --save_path results/erased/tyler_edlin \
  2>&1 | tee logs/erased_tyler_edlin.log &
PID3=$!
echo "  [3/6] Tyler Edlin     — PID $PID3  (logs/erased_tyler_edlin.log)"

python generate_esd_old.py \
  --esd_path "$WEIGHTS/diffusers-KilianEng-ESDx1-UNET.pt" \
  --prompts_path data/short_niche_art_prompts.csv \
  --artist "Kilian Eng" \
  --save_path results/erased/kilian_eng \
  2>&1 | tee logs/erased_kilian_eng.log &
PID4=$!
echo "  [4/6] Kilian Eng      — PID $PID4  (logs/erased_kilian_eng.log)"

python generate_esd_old.py \
  --esd_path "$WEIGHTS/diffusers-ThomasKinkade-ESDx1-UNET.pt" \
  --prompts_path data/short_niche_art_prompts.csv \
  --artist "Thomas Kinkade" \
  --save_path results/erased/thomas_kinkade \
  2>&1 | tee logs/erased_thomas_kinkade.log &
PID5=$!
echo "  [5/6] Thomas Kinkade  — PID $PID5  (logs/erased_thomas_kinkade.log)"

python generate_esd_old.py \
  --esd_path "$WEIGHTS/diffusers-AjinDemiHuman-ESDx1-UNET.pt" \
  --prompts_path data/short_niche_art_prompts.csv \
  --artist "Ajin: Demi Human" \
  --save_path results/erased/ajin \
  2>&1 | tee logs/erased_ajin.log &
PID6=$!
echo "  [6/6] Ajin: Demi Human — PID $PID6  (logs/erased_ajin.log)"

echo ""
echo "  All 6 jobs running. Monitor with:"
echo "    tail -f logs/erased_kelly.log"
echo "    watch -n 5 'ls results/erased/*/  | grep png | wc -l'"
echo ""

wait

echo ""
echo "============================================================"
echo "  ALL 6 ERASED JOBS COMPLETE"
echo "  Images saved to: results/erased/"
echo "============================================================"
