#!/bin/bash
# Run the fair Van Gogh comparison table and chart over the same prompt pairs.
# CA is included for prompt-pair metrics but excluded from FID/KID by design.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/baselines/use_venv.sh"

EXPECTED_PAIRS="${EXPECTED_PAIRS:-250}"
EXPECTED_FID_IMAGES="${EXPECTED_FID_IMAGES:-2000}"
METHOD_KEYS="${METHOD_KEYS:-esd,uce,ca,space,space_v3}"
FID_METHOD_KEYS="${FID_METHOD_KEYS:-esd,uce,space,space_v3}"
GRID_METHODS="${GRID_METHODS:-ESD-x,UCE,CA,SPACE,SPACE-v3}"

cd "$ROOT"
mkdir -p results/evaluation results/comparisons

count_pngs() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f -name "*.png" | wc -l | tr -d " "
}

echo "============================================================"
echo "  Fair Van Gogh evaluation"
echo "============================================================"
echo "Expected prompt pairs per method: $EXPECTED_PAIRS"
echo "Methods: $METHOD_KEYS"
echo "FID/KID methods: $FID_METHOD_KEYS"
echo "CA FID/KID: disabled"
echo ""

"$PYTHON_BIN" - <<PY
import math
from pathlib import Path

import pandas as pd

expected = int("$EXPECTED_PAIRS")
df = pd.read_csv("data/vangogh_prompts.csv")
if "artist" in df.columns:
    df = df[df["artist"].astype(str).str.strip().str.lower() == "vincent van gogh"]
prompt_count = len(df)
if prompt_count <= 0:
    raise SystemExit("No Van Gogh prompts found in data/vangogh_prompts.csv")
if expected % prompt_count != 0:
    raise SystemExit(f"EXPECTED_PAIRS={expected} is not divisible by prompt_count={prompt_count}")
copies = expected // prompt_count

case_dirs = {
    "Vanilla": Path("results/baseline/vangogh"),
    "ESD-x": Path("results/erased/vangogh"),
    "UCE": Path("results/uce/uce-Van_Gogh"),
    "SPACE": Path("results/space/space-Van_Gogh"),
    "SPACE-v3": Path("results/space_v3/space-Van_Gogh"),
}
for label, folder in case_dirs.items():
    missing = []
    for case in df["case_number"].astype(int).tolist():
        for idx in range(copies):
            if not (folder / f"{case}_{idx}.png").exists():
                missing.append(f"{case}_{idx}.png")
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break
    if missing:
        raise SystemExit(f"{label} is missing prompt-pair images in {folder}: {missing[:5]}")
    print(f"OK: {label} has {expected} matched prompt-pair images")

ca_dir = Path("results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples")
ca_needed = [ca_dir / f"{idx:05d}.png" for idx in range(expected)]
ca_missing = [p.name for p in ca_needed if not p.exists()]
if ca_missing:
    raise SystemExit(f"CA is missing ordered prompt-pair images in {ca_dir}: {ca_missing[:5]}")
print(f"OK: CA has {expected} ordered prompt-pair images")

fid_expected = int("$EXPECTED_FID_IMAGES")
fid_keys = [key.strip() for key in "$FID_METHOD_KEYS".split(",") if key.strip()]
fid_dirs = {"esd": "erased", "uce": "uce", "space": "space", "space_v3": "space_v3"}
baseline_count = len(list(Path("results/fid/baseline").glob("*.png")))
if baseline_count != fid_expected:
    raise SystemExit(f"FID baseline has {baseline_count} images; expected {fid_expected}")
print(f"OK: FID baseline has {fid_expected} images")
for key in fid_keys:
    folder_name = fid_dirs.get(key)
    if folder_name is None:
        raise SystemExit(f"No FID folder mapping is defined for method key {key}")
    folder = Path("results/fid") / folder_name
    n = len(list(folder.glob("*.png")))
    if n != fid_expected:
        raise SystemExit(f"FID folder for {key} ({folder}) has {n} images; expected {fid_expected}")
    print(f"OK: FID {key} has {fid_expected} images")
PY

rm -f results/evaluation/metrics_cache_v5_van_gogh.json

"$PYTHON_BIN" evaluate.py \
  --only-artist "Van Gogh" \
  --method-keys "$METHOD_KEYS" \
  --expected-pairs "$EXPECTED_PAIRS" \
  --fid-method-keys "$FID_METHOD_KEYS" \
  --disable-fid-fallback

cp results/evaluation/metrics.csv results/evaluation/fair_vangogh_metrics.csv
cp results/evaluation/ablation_table.tex results/evaluation/fair_vangogh_ablation_table.tex
cp results/evaluation/comparison_bars.png results/evaluation/fair_vangogh_comparison_bars.png

"$PYTHON_BIN" make_comparison_grids.py \
  --only-artist "Van Gogh" \
  --methods "$GRID_METHODS"

cp results/comparisons/van_gogh.png results/comparisons/fair_vangogh.png
cp results/comparisons/van_gogh.pdf results/comparisons/fair_vangogh.pdf

echo ""
echo "Fair Van Gogh evaluation complete."
echo "Metrics: results/evaluation/fair_vangogh_metrics.csv"
echo "Table: results/evaluation/fair_vangogh_ablation_table.tex"
echo "Bars: results/evaluation/fair_vangogh_comparison_bars.png"
echo "Grid: results/comparisons/fair_vangogh.png"
