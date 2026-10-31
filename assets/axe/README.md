# Default axe example

- `mesh.glb`: untextured input mesh with 50,000 triangles, simplified from the original 1,000,000-triangle asset.
- `image.png`: 1024 × 1024 reference image.

These are the inputs used for the demonstrated albedo → MR → UV baking/inpainting run. The input mesh has no UV map; inference generates its UVs automatically. No simplification tool is needed to run this prepared example.

From the repository root, after installing the environment and placing the model weights:

```bash
bash scripts/run_miku.sh
```

The default configuration uses 1024 × 1024 geometry views, 2048 × 2048 UV textures, pack-mode MR and geometry-mask background whitening. Outputs are written to `outputs/axe/`, including `textured_mesh.glb`.

See the main [inference guide](../../README.md#inference) for custom inputs and settings.
