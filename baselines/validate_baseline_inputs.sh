#!/bin/bash
# Validate and repair lightweight tracked inputs before expensive baseline runs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

progress_banner "Validate baseline inputs"

repair_tracked_file() {
  local rel="$1"
  local path="$ROOT/$rel"
  if [ -f "$path" ]; then
    echo "[$(progress_now)] OK: $rel"
    return 0
  fi

  echo "[$(progress_now)] Missing tracked input: $rel"
  if git -C "$ROOT" ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
    echo "[$(progress_now)] Restoring $rel from HEAD"
    git -C "$ROOT" checkout -- "$rel"
  fi

  if [ ! -f "$path" ]; then
    echo "ERROR: required input is still missing after repair: $path"
    echo "Repo root: $ROOT"
    echo "Current branch/commit:"
    git -C "$ROOT" status --short --branch || true
    exit 1
  fi
}

progress_step 1 3 "checking prompt CSV files"
repair_tracked_file "data/kelly_prompts.csv"
repair_tracked_file "data/vangogh_prompts.csv"
repair_tracked_file "data/short_niche_art_prompts.csv"

progress_step 2 3 "checking preservation CSV"
repair_tracked_file "data/coco_30k.csv"

progress_step 3 3 "validating CSV schemas"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path.cwd()
required = {"case_number", "prompt", "evaluation_seed"}
files = [
    "data/kelly_prompts.csv",
    "data/vangogh_prompts.csv",
    "data/short_niche_art_prompts.csv",
]

for rel in files:
    path = root / rel
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{rel} missing columns: {missing}")
    if df.empty:
        raise SystemExit(f"{rel} has no rows")
    print(f"OK: {rel} rows={len(df)}")

coco = root / "data/coco_30k.csv"
df = pd.read_csv(coco)
if "prompt" not in df.columns:
    raise SystemExit("data/coco_30k.csv missing prompt column")
if df.empty:
    raise SystemExit("data/coco_30k.csv has no rows")
print(f"OK: data/coco_30k.csv rows={len(df)}")
PY

echo "[$(progress_now)] Baseline inputs OK."
