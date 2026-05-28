#!/bin/bash
# Shared progress helpers for long RunPod baseline jobs.

set -euo pipefail

progress_now() {
  date +"%Y-%m-%d %H:%M:%S"
}

progress_banner() {
  local label="$1"
  echo ""
  echo "============================================================"
  echo "[$(progress_now)] $label"
  echo "============================================================"
}

progress_step() {
  local current="$1"
  local total="$2"
  local label="$3"
  local width=24
  local filled=$((current * width / total))
  local empty=$((width - filled))
  local bar=""
  local i
  for ((i = 0; i < filled; i++)); do bar="${bar}#"; done
  for ((i = 0; i < empty; i++)); do bar="${bar}-"; done
  echo "[$(progress_now)] [$bar] $current/$total $label"
}

run_with_progress() {
  local label="$1"
  shift
  local interval="${PROGRESS_INTERVAL:-30}"
  local start
  start="$(date +%s)"
  echo "[$(progress_now)] START: $label"
  (
    while true; do
      sleep "$interval"
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start))
      echo "[$(progress_now)] STILL RUNNING: $label (${elapsed}s elapsed)"
    done
  ) &
  local heartbeat_pid="$!"
  set +e
  "$@"
  local code="$?"
  set -e
  kill "$heartbeat_pid" >/dev/null 2>&1 || true
  wait "$heartbeat_pid" >/dev/null 2>&1 || true
  local end elapsed
  end="$(date +%s)"
  elapsed=$((end - start))
  if [ "$code" -eq 0 ]; then
    echo "[$(progress_now)] DONE: $label (${elapsed}s)"
  else
    echo "[$(progress_now)] FAILED: $label (${elapsed}s, exit $code)"
  fi
  return "$code"
}
