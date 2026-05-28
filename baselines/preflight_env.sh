#!/bin/bash
# Fast environment preflight for baseline scripts. Use bootstrap_runpod_env.sh to install/repair.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
READY_MARKER="$VENV_DIR/.space_baseline_env_ready"
source "$ROOT/baselines/progress.sh"
source "$ROOT/baselines/use_venv.sh"

progress_banner "Preflight baseline environment"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "Warning: no active virtualenv detected. Recommended:"
  echo "  source $VENV_DIR/bin/activate"
fi

ensure_pip() {
  if python -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  if [ "${AUTO_REPAIR_ENV:-0}" = "1" ]; then
    echo "pip is missing/broken; attempting bootstrap via ensurepip..."
    python -m ensurepip --upgrade
    python -m pip --version >/dev/null 2>&1 && return 0
  fi
  echo "pip is missing/broken. Run:"
  echo "  bash baselines/bootstrap_runpod_env.sh"
  exit 1
}

verify_env() {
  python - <<'PY'
import importlib
import sys

checks = [
    ("setuptools", "69.5.1"),
    ("packaging", "24.2"),
    ("torch", "2.1.2"),
    ("torchvision", "0.16.2"),
    ("numpy", "1.26.4"),
    ("diffusers", "0.21.4"),
    ("transformers", "4.35.0"),
    ("accelerate", "0.24.1"),
    ("wandb", "0.17.9"),
    ("pytorch_lightning", "1.9.0"),
    ("torchmetrics", "0.11.1"),
    ("omegaconf", "2.1.1"),
    ("einops", "0.3.0"),
    ("kornia", "0.6"),
]
imports = [
    "matplotlib",
    "lpips",
    "pandas",
    "PIL",
    "safetensors",
    "tqdm",
    "scipy.signal",
    "torch_fidelity.feature_extractor_inceptionv3",
    "pkg_resources",
    "packaging",
]
errors = []
for module, expected_prefix in checks:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "")
        if not str(version).startswith(expected_prefix):
            errors.append(f"{module}: expected {expected_prefix}*, found {version}")
    except Exception as exc:
        errors.append(f"{module}: import failed ({exc})")
for module in imports:
    try:
        importlib.import_module(module)
    except Exception as exc:
        errors.append(f"{module}: import failed ({exc})")
if errors:
    print("Baseline environment check failed:")
    for error in errors:
        print(f" - {error}")
    sys.exit(1)
PY
}

ensure_pip

progress_step 1 2 "import/version checks"
if verify_env; then
  progress_step 2 2 "environment healthy"
  if [ -f "$READY_MARKER" ]; then
    echo "Preflight OK: ready marker present."
  else
    echo "Preflight OK: environment works, but ready marker is missing."
    echo "Run bootstrap once to create it:"
    echo "  bash baselines/bootstrap_runpod_env.sh"
  fi
  python - <<'PY'
import torch, diffusers, transformers, numpy
print("torch:", torch.__version__)
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
  exit 0
fi

if [ "${AUTO_REPAIR_ENV:-0}" = "1" ]; then
  echo "Attempting environment repair through bootstrap..."
  run_with_progress "repair baseline environment" env AUTO_REPAIR_ENV=0 REPAIR_REQUIREMENTS=1 FORCE_ENV_REBUILD="${FORCE_ENV_REBUILD:-0}" bash "$ROOT/baselines/bootstrap_runpod_env.sh" || \
  run_with_progress "rebuild baseline environment" env AUTO_REPAIR_ENV=0 FORCE_ENV_REBUILD=1 bash "$ROOT/baselines/bootstrap_runpod_env.sh"
  exit 0
fi

echo ""
echo "Run this once to repair/install the persistent environment:"
echo "  bash baselines/bootstrap_runpod_env.sh"
exit 1
