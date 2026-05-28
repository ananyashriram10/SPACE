#!/bin/bash
# One-time persistent RunPod venv bootstrap for official baselines.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
READY_MARKER="$VENV_DIR/.space_baseline_env_ready"
REQ="$ROOT/baselines/requirements-runpod-baselines.txt"
source "$ROOT/baselines/progress.sh"

progress_banner "Bootstrap RunPod baseline environment"
echo "Repo: $ROOT"
echo "Venv: $VENV_DIR"
echo "Requirements: $REQ"

if [ "${FORCE_ENV_REBUILD:-0}" = "1" ] && [ -d "$VENV_DIR" ]; then
  backup="$ROOT/.venv.replaced.$(date +%Y%m%d_%H%M%S)"
  progress_step 1 7 "replacing existing venv"
  echo "Moving old venv to: $backup"
  rm -f "$READY_MARKER"
  mv "$VENV_DIR" "$backup"
fi

if [ ! -d "$VENV_DIR" ]; then
  progress_step 1 7 "creating venv"
  run_with_progress "python3 -m venv $VENV_DIR" python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

progress_step 2 7 "bootstrapping pip"
run_with_progress "ensurepip" python -m ensurepip --upgrade

progress_step 3 7 "pinning packaging tools"
run_with_progress "install pinned pip/setuptools/packaging/wheel" python -m pip install --progress-bar on \
  "pip==24.3.1" \
  "setuptools==69.5.1" \
  "packaging==24.2" \
  "wheel==0.44.0"

if [ "${FORCE_ENV_REBUILD:-0}" = "1" ]; then
  rm -f "$READY_MARKER"
fi

progress_step 4 7 "checking torch stack"
torch_ok=0
python - <<'PY' >/dev/null 2>&1 && torch_ok=1 || torch_ok=0
import torch, torchvision
assert torch.__version__.startswith("2.1.2")
assert torchvision.__version__.startswith("0.16.2")
PY

if [ "$torch_ok" != "1" ] || [ "${FORCE_TORCH_PIN:-0}" = "1" ]; then
  progress_step 5 7 "installing pinned torch stack"
  run_with_progress "install torch==2.1.2 torchvision==0.16.2 cu118" python -m pip install --progress-bar on --no-cache-dir --ignore-installed \
    "torch==2.1.2" \
    "torchvision==0.16.2" \
    --index-url "https://download.pytorch.org/whl/cu118"
else
  echo "[$(progress_now)] Torch stack already pinned."
fi

if [ ! -f "$READY_MARKER" ] || [ "${REPAIR_REQUIREMENTS:-0}" = "1" ]; then
  progress_step 6 7 "installing baseline requirements"
  if [ "${FORCE_ENV_REBUILD:-0}" = "1" ]; then
    run_with_progress "force reinstall baseline requirement wheels without deps" python -m pip install --progress-bar on --force-reinstall --no-deps -r "$REQ"
  fi
  run_with_progress "install pinned baseline requirements" python -m pip install --progress-bar on -r "$REQ"
else
  echo "[$(progress_now)] Ready marker exists; skipping requirement install."
fi

progress_step 7 7 "verifying environment"
if run_with_progress "preflight verification" env AUTO_REPAIR_ENV=0 bash "$ROOT/baselines/preflight_env.sh"; then
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$READY_MARKER"
  echo "Baseline environment ready: $READY_MARKER"
else
  rm -f "$READY_MARKER"
  exit 1
fi
