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
