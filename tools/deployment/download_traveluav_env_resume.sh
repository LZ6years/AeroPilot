#!/usr/bin/env bash
# Replacement launcher for the installed HF 1.33.0; stop the old CLI first.
set -euo pipefail
DEST="${1:-$HOME/TravelUAV/ENVs}"
if (( $# > 0 )); then shift; fi
PY="${HF_PYTHON:-$HOME/miniconda3/bin/python}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FILTER=()
if (( $# > 0 )); then FILTER=(--include "$@"); fi
trap 'echo "Stopped; run the same command to resume."; exit 130' INT TERM
for ((attempt=1; attempt<=20; attempt++)); do
  if "$PY" "$SCRIPT_DIR/hf_resume_download.py" --local-dir "$DEST" "${FILTER[@]}"; then
    exit 0
  fi
  if (( attempt < 20 )); then
    echo "Attempt $attempt failed. Retry in 30 seconds."
    sleep 30
  fi
done
exit 1
