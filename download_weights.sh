#!/bin/bash
# ============================================================
# download_weights.sh — Download all 6 pre-trained ESD-x weights
# in parallel using wget (6 files × 3.2 GB = ~19 GB total).
#
# Run inside tmux so you can detach and come back.
# ============================================================

set -e
mkdir -p esd-weights/art
BASE="https://erasing.baulab.info/weights/esd_models/art"

FILES=(
  "diffusers-KellyMcKernan-ESDx1-UNET.pt"
  "diffusers-VanGogh-ESDx1-UNET.pt"
  "diffusers-TylerEdlin-ESDx1-UNET.pt"
  "diffusers-KilianEng-ESDx1-UNET.pt"
  "diffusers-ThomasKinkade-ESDx1-UNET.pt"
  "diffusers-AjinDemiHuman-ESDx1-UNET.pt"
)

echo "============================================================"
echo "  Downloading 6 ESD-x checkpoints (~19 GB total)"
echo "  Using wget with 6 parallel downloads"
echo "============================================================"
echo ""

PIDS=()
for f in "${FILES[@]}"; do
  OUT="esd-weights/art/$f"
  # Skip if already fully downloaded
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT" 2>/dev/null || stat -f%z "$OUT")" -gt 3000000000 ]; then
    echo "  SKIP (exists): $f"
    continue
  fi
  echo "  Starting: $f"
  wget -q --show-progress -c -O "$OUT" "$BASE/$f" &
  PIDS+=($!)
done

echo ""
echo "  All downloads running in parallel. Waiting..."
echo ""

# Wait for all and report
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  ERROR: a download failed (PID $pid)"
    FAILED=1
  fi
done

echo ""
echo "============================================================"
if [ $FAILED -eq 0 ]; then
  echo "  All weights downloaded to esd-weights/art/"
  ls -lh esd-weights/art/
else
  echo "  Some downloads failed. Re-run this script to resume."
fi
echo "============================================================"
