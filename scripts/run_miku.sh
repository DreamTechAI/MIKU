#!/usr/bin/env bash
set -euo pipefail
MIKU_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MIKU_PYTHON=${MIKU_PYTHON:-python}
export PYTHONPATH="$MIKU_ROOT/custom_rasterizer:$MIKU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$MIKU_ROOT"
exec "$MIKU_PYTHON" infer_miku.py "$@"
