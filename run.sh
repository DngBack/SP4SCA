#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config.yaml}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-codex}"
mkdir -p "${MPLCONFIGDIR}"

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

"${PY_BIN}" scripts/run_pipeline.py --config "${CONFIG_PATH}"
