#!/bin/sh
# Double-click on macOS, or run from a terminal, to open the standalone viewer.
cd "$(dirname "$0")/.." || exit 1
exec "${MIKU_PYTHON:-python3}" tools/preview_glb.py --port 0 "$@"
