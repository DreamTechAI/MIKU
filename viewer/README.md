# MIKU Material Studio

A local-first GLB viewer for presenting MIKU's exported PBR materials. The only
bundled demo is the generated Dragonbone Axe: 50,000 triangles, 2K albedo and metallic /
roughness textures. It is a real-time 3D model, not a prerendered image.

## Open the studio

From the repository root:

```bash
python3 tools/preview_glb.py
```

The browser opens at `http://127.0.0.1:8765` with the bundled generated axe.
Keep the terminal open; press **Ctrl+C** to stop the viewer. On macOS, you can
also double-click [`scripts/open_studio.command`](../scripts/open_studio.command).

You only need **Python 3.9+ and a browser with WebGL 2**. Enable browser hardware acceleration.
No model weights, inference environment, or additional Python or Node packages
are needed.

Drag a self-contained `.glb` onto the page, or choose **打开模型**.
Imported models are parsed in the browser and are not
uploaded. All rendering code and decoders are bundled locally. The interface uses Clash Display
from the official Fontshare CDN, with a system-font fallback when offline; rendering
and local GLB imports work without that font request.

```bash
# Open an existing exported model
python3 tools/preview_glb.py --model outputs/axe/textured_mesh.glb

# Open the result when inference finishes
bash scripts/run_miku.sh --preview

# Select an available port automatically
python3 tools/preview_glb.py --port 0
```

On a headless server, the helper prints an SSH port-forwarding command. Run that
command on the desktop, then open the printed URL. You can also copy the GLB to
your desktop and drag it into an already open viewer. The server binds to
localhost and serves only the viewer assets and the selected GLB.

The interface follows a Midnight Editorial style: near-black surfaces, oversized
Clash Display typography, fine dividers, and restrained reveal motion. Reduced-motion
preferences are respected. The font is served by [Fontshare](https://www.fontshare.com/fonts/clash-display);
font files are not redistributed with this repository.

## Presentation controls

- **摄影影棚 / 柔光画廊 / 暖金轮廓**: three HDR studio lighting treatments.
- **摄影渲染 (PT)**: progressive GPU path tracing with six light bounces, real
  indirect lighting and rough floor reflections. Keep the camera still to accumulate
  up to 128 samples; rotating restarts convergence. The interactive path-traced image is capped at a
  1600-pixel long edge for responsiveness; exported images render at full 4K. This mode requires the bundled
  HDRI and supports models up to 500,000 triangles. Physical shadows remain enabled.
- Adjust exposure and lighting rotation without altering the exported material.
- Switch between full PBR, unlit albedo, geometry normals, and a neutral clay
  material. Normal mode is a geometry visualization, not a generated normal
  texture.
- Real planar reflections, soft contact shadows, model rotation and up-axis.
- Drag to orbit, scroll to zoom, right-drag to pan. `R` resets the view, `Space`
  toggles rotation, and `F` enters or exits a clean presentation view.
- **导出展示图** saves the current view as a PNG, with a 3840-pixel long edge
  (limited by the browser's maximum texture size). UI controls are excluded. In photography mode the 4K output accumulates 128
  samples; progress is shown and the export can be cancelled. This is more expensive
  than real-time mode and can take a while depending on the GPU. After export,
  the viewer returns to real-time mode to release the GPU.

The viewer preserves the GLB's material maps. If a static mesh omits vertex normals,
it reconstructs smooth normals with 60-degree creases for continuous highlights.
Existing authored normals are preserved. It does not invent texture detail,
reconstruct geometry, or rerun inference. Embedded PNG/JPEG, Draco-compressed
geometry and Meshopt-compressed geometry are supported. Export embedded PNG/JPEG
textures instead of KTX2/BasisU. External resource URLs are not loaded. Animation
playback and editing are outside this viewer's scope. Very large assets may need
mesh simplification or smaller textures; the file picker limit is 512 MB.

## Source and dependencies

`dist/` contains the editable HTML, CSS and JavaScript source and is ready to serve
without a build step. `dist/vendor/` contains the required subset of **Three.js
0.180.0**, from the official `three` npm package, including its GLTF, Draco and
Meshopt loaders. Photography mode bundles `three-gpu-pathtracer 0.0.24` and
`three-mesh-bvh 0.9.5`, both MIT-licensed. `dist/assets/axe.glb` is the generated example output, not a model
checkpoint. No inference weights are included.

Studio lighting uses the bundled [Studio Small 09](https://polyhaven.com/a/studio_small_09)
HDRI by Sergej Majboroda / Poly Haven, distributed under
[CC0](https://polyhaven.com/license). Procedural softboxes serve as a fallback.
Neutral and AgX tone mapping, exposure and environment intensity are adjustable.

Third-party licenses are included under `dist/vendor/`: Three.js, three-gpu-pathtracer, three-mesh-bvh (MIT),
Meshoptimizer (MIT), Draco (Apache 2.0), and the path tracer's glslSmartDeNoise
shader by Michele Morrone (BSD 2-Clause). Vendor files are unmodified.

The same `dist/` directory can be served by an ordinary static web host. Use HTTP
or HTTPS rather than opening `index.html` as a `file://` URL, because the browser
loads ES modules and the GLB via HTTP.
