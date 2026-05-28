#!/bin/bash
# Activate the repo-local baseline venv when available.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [ -f "$VENV_DIR/bin/activate" ] && [ "${VIRTUAL_ENV:-}" != "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$VENV_DIR/bin/python" ]; then
    export PYTHON_BIN="$VENV_DIR/bin/python"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="python"
  else
    export PYTHON_BIN="python3"
  fi
fi
