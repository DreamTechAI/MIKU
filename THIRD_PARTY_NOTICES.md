# Third-party acknowledgments

MIKU builds on [LumiTex](https://github.com/LumiTexPBR/LumiTex),
[UniTEX](https://github.com/YixunLiang/UniTEX), and
[MaterialMVP](https://github.com/ZebinHe/MaterialMVP).

The renderer, UV baking and inpainting components are adapted from MaterialMVP.
The model pipelines extend [Hugging Face Diffusers](https://github.com/huggingface/diffusers).
Upstream authorship and license notices remain with the corresponding components;
this document does not replace their terms or grant rights to third-party weights.

`miku/vendor/dinov2/` contains an inference subset of
[Meta DINOv2](https://github.com/facebookresearch/dinov2), licensed under Apache 2.0.
Its license is included at `miku/vendor/dinov2/LICENSE`.
Local modifications use relative imports and retain only tensor forward computation,
omitting stochastic-depth and nested training paths. No DINO training head, loss,
dataset, trainer or model checkpoint is included.

Base-model and adapter weights are distributed separately and retain their own
licenses and access conditions.

`viewer/dist/vendor/` contains an unmodified subset of Three.js 0.180.0 (MIT),
with bundled Draco (Apache 2.0) and Meshoptimizer (MIT) decoders. The corresponding
license files are included in that directory. The studio includes Studio Small 09 by Sergej Majboroda / Poly Haven
(https://polyhaven.com/a/studio_small_09), distributed under CC0
(https://polyhaven.com/license). Procedural lighting is also included as a fallback.

Photography mode bundles three-gpu-pathtracer 0.0.24 and three-mesh-bvh 0.9.5
by Garrett Johnson, both licensed under MIT. The license files are included in
`viewer/dist/vendor/`.

The path tracer also includes glslSmartDeNoise by Michele Morrone (BSD 2-Clause).
Its upstream license is included at `viewer/dist/vendor/SMARTDENOISE-LICENSE.txt`;
the shader's original copyright notice is retained in `pathtracer.module.js`.
Source: https://github.com/BrutPitt/glslSmartDeNoise/.
