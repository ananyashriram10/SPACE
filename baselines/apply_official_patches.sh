#!/bin/bash
# Apply documented minimal compatibility patches to official baseline repos.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$ROOT/baselines/patches"
UCE_DIR="$ROOT/baselines/external/unified-concept-editing"
UCE_ERASE_SCRIPT="$UCE_DIR/trainscripts/uce_sd_erase.py"
CA_FILTER="$ROOT/baselines/external/concept-ablation/compvis/src/filter.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python interpreter found. Activate .venv or install python3."
    exit 1
  fi
fi

if [ ! -d "$UCE_DIR/.git" ]; then
  echo "Missing official UCE repo at $UCE_DIR"
  echo "Run: bash baselines/setup_official_repos.sh"
  exit 1
fi

patch_uce_vae_runtime_fix() {
  if [ ! -f "$UCE_ERASE_SCRIPT" ]; then
    echo "Missing UCE erase script: $UCE_ERASE_SCRIPT"
    exit 1
  fi

  if ! grep -q "vae=None" "$UCE_ERASE_SCRIPT"; then
    echo "Patch already applied (no vae=None): unified-concept-editing_7c724d9_vae_fix.patch"
    return 0
  fi

  if git -C "$UCE_DIR" apply --check "$PATCH_DIR/unified-concept-editing_7c724d9_vae_fix.patch" >/dev/null 2>&1; then
    git -C "$UCE_DIR" apply "$PATCH_DIR/unified-concept-editing_7c724d9_vae_fix.patch"
    echo "Applied patch: unified-concept-editing_7c724d9_vae_fix.patch"
  else
    # Fallback for minor context drift while preserving closing syntax.
    "$PYTHON_BIN" - <<PY
import re
from pathlib import Path
p = Path("${UCE_ERASE_SCRIPT}")
text = p.read_text()
text2 = re.sub(r",\s*\n\s*vae=None\)\.to\(device\)", ").to(device)", text, count=1)
p.write_text(text2)
PY
    echo "Applied fallback inline fix: removed vae=None arg while preserving call syntax"
  fi

  if grep -q "vae=None" "$UCE_ERASE_SCRIPT"; then
    echo "Failed to apply UCE vae runtime fix."
    exit 1
  fi

  "$PYTHON_BIN" -m py_compile "$UCE_ERASE_SCRIPT" >/dev/null 2>&1 || {
    echo "Patched UCE script is not syntactically valid: $UCE_ERASE_SCRIPT"
    exit 1
  }
}

patch_uce_vae_runtime_fix

# Patch: lazy-load SSCD model in CA compvis/src/filter.py
# The original code loads the SSCD torchscript model at MODULE IMPORT TIME
# which crashes immediately even for style/instance ablation (which never calls
# filter() at all — SSCD filtering is only used for memorization ablation).
# This patch converts it to a lazy singleton that only loads on first call.
patch_ca_filter_lazy_sscd() {
  if [ ! -f "$CA_FILTER" ]; then
    echo "CA filter.py not found at $CA_FILTER — skipping SSCD lazy-load patch"
    return 0
  fi

  # Idempotency check
  if grep -q '_sscd_model_cache' "$CA_FILTER"; then
    echo "Patch already applied: ca_filter_lazy_sscd"
    return 0
  fi

  CA_FILTER="$CA_FILTER" "$PYTHON_BIN" - <<'PY'
import re, os, sys
from pathlib import Path

p = Path(os.environ["CA_FILTER"])
text = p.read_text()

old = 'model = torch.jit.load(\n    "../assets/pretrained_models/sscd_imagenet_mixup.torchscript.pt")'
new = '''_sscd_model_cache = None


def _get_sscd_model():
    """Lazy-load SSCD only when needed (memorization ablation). Style/instance ablation never calls filter()."""
    global _sscd_model_cache
    if _sscd_model_cache is None:
        _sscd_model_cache = torch.jit.load(
            "../assets/pretrained_models/sscd_imagenet_mixup.torchscript.pt")
    return _sscd_model_cache'''

if old not in text:
    print("Could not find eager model load \u2014 skipping (code may have changed upstream)")
    sys.exit(0)

text = text.replace(old, new)
# Replace direct model(batch) calls with lazy getter
text = re.sub(r'\bmodel\(batch\)', '_get_sscd_model()(batch)', text)
p.write_text(text)
print("Applied patch: ca_filter_lazy_sscd (SSCD now lazy-loaded)")
PY
}

patch_ca_filter_lazy_sscd

# Patch: shim retrieve_timesteps for diffusers < 0.27 in CA diffusers model_pipeline.py
CA_DIFFUSERS_PIPELINE="$ROOT/baselines/external/concept-ablation/diffusers/model_pipeline.py"
patch_ca_retrieve_timesteps() {
  if [ ! -f "$CA_DIFFUSERS_PIPELINE" ]; then
    echo "CA diffusers model_pipeline.py not found — skipping retrieve_timesteps patch"
    return 0
  fi
  if ! grep -q "retrieve_timesteps" "$CA_DIFFUSERS_PIPELINE"; then
    echo "Patch already applied: ca_retrieve_timesteps_shim"
    return 0
  fi
  "$PYTHON_BIN" - "$CA_DIFFUSERS_PIPELINE" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text()
old = "from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg, StableDiffusionPipelineOutput"
new = """try:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, rescale_noise_cfg, StableDiffusionPipelineOutput
except ImportError:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, StableDiffusionPipelineOutput
    def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, **kwargs):
        if timesteps is not None:
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)"""
if old in text:
    text = text.replace(old, new)
    p.write_text(text)
    print("Applied patch: ca_retrieve_timesteps_shim")
else:
    print("retrieve_timesteps import not found in expected form — skipping")
PY
}

patch_ca_retrieve_timesteps

echo "Official compatibility patches ready."
