#!/usr/bin/env bash
# Build the two local inference extensions in this checkout.
set -euo pipefail
MIKU_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MIKU_PYTHON=${MIKU_PYTHON:-python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$(dirname -- "$MIKU_PYTHON"):$CUDA_HOME/bin:$PATH"
# PyTorch detects the visible GPU architecture; set TORCH_CUDA_ARCH_LIST to override.
export MAX_JOBS=${MAX_JOBS:-4}
mkdir -p "$MIKU_ROOT/work"
cd "$MIKU_ROOT/custom_rasterizer"
"$MIKU_PYTHON" setup.py build_ext --inplace 2>&1 | tee "$MIKU_ROOT/work/build_rasterizer.log"
cd "$MIKU_ROOT/DifferentiableRenderer"
MIKU_EXT_SUFFIX=$("$MIKU_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
# pybind11 emits a list of compiler -I arguments.
read -r -a MIKU_INCLUDE_FLAGS <<< "$("$MIKU_PYTHON" -m pybind11 --includes)"
c++ -O3 -Wall -shared -std=c++11 -fPIC "${MIKU_INCLUDE_FLAGS[@]}" mesh_inpaint_processor.cpp \
    -o "mesh_inpaint_processor$MIKU_EXT_SUFFIX" 2>&1 | tee "$MIKU_ROOT/work/build_inpaint.log"
