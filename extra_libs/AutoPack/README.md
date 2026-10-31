# AutoPack

Standalone C++17/pybind11 rectangle-packing extension used by the original MIKU
tools. The module name is case-sensitive: `import AutoPack`.

## Installation

From the MIKU repository root, with the environment activated:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps ./extra_libs/AutoPack
python -c "import AutoPack; print(AutoPack.__file__)"
```

A C++17 compiler and Python development headers are required. The build uses
`setuptools`, `wheel` and `pybind11`, already listed in the root requirements.
AutoPack itself runs on the CPU and requires no CUDA toolkit or model weights.
Only source is included; the extension is compiled for the active Python
interpreter during installation.

## Usage

```python
import AutoPack

sizes = [(32, 48), (48, 32), (16, 32), (32, 16), (16, 16), (32, 32)]
order, utilization, layout = AutoPack.pack_main_loop(sizes, allow_rotate=False)
print(layout.width, layout.height, utilization)
for rect in layout.used_rects:
    print(rect.rid, rect.x, rect.y, rect.w, rect.h, rect.rotated)
```

Pass a nonempty sequence of positive `(width, height)` pairs in a consistent
unit. `rect.rid` identifies the original zero-based input index. The extension
returns rectangle placement metadata; callers perform image cropping, pasting
and inverse mapping themselves. `allow_rotate` defaults to `True`; pass `False`
when the caller does not support rotating and restoring patches. Its genetic
search is stochastic, so repeated calls may choose different valid layouts.

The current default MIKU inference path uses the bundled Python packing functions
in [`utils/miku_multiview_utils.py`](../../utils/miku_multiview_utils.py).
This extension is supplied separately for callers of `AutoPack.pack_main_loop`;
installing it does not change the default inference packing implementation.
