<div align="center">

# MIKU
### Material Invariance and Knowledge Unification for PBR Generation

**Yajie Bao**<sup>1,2</sup>, **Jiaye Fu**<sup>3,2</sup>, **Youtian Lin**<sup>1</sup>, **Shuang Wu**<sup>1,2</sup>, **Jiachen Qian**<sup>4,2</sup>, **Feihu Zhang**<sup>2</sup>, **Xun Cao**<sup>1</sup>, **Yao Yao**<sup>1</sup>

<sup>1</sup> Nanjing University · <sup>2</sup> DreamTech · <sup>3</sup> Peking University · <sup>4</sup> The University of Hong Kong

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](docs/paper.pdf)
[![Project](https://img.shields.io/badge/Project-Page-70bfa3)](docs/index.html)
[![DOI](https://img.shields.io/badge/Publication-DOI-blue)](https://doi.org/10.1145/3842521)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E)](https://huggingface.co/datasets/linxizhilll/MIKU)

**Congratulations 🎉 Accepted to SIGGRAPH Asia 2026!**

</div>

![MIKU generated scene](docs/assets/teaser.webp)

MIKU generates multi-view albedo from a reference image and mesh geometry, predicts metallic–roughness maps, then bakes and inpaints the views into UV textures and a PBR GLB.

## News

- **[2026-10-31]** Inference code is available. Model weights and download links will be provided separately.

## Installation

Inference requires **Linux x86_64, Python 3.10, CUDA 12.4, PyTorch 2.6.0** and an NVIDIA GPU.

Install the NVIDIA driver and **CUDA toolkit 12.4** (including `nvcc`), a C++ compiler (GCC/G++ 11) and Python development headers before building. Python dependencies, including the CUDA-enabled PyTorch wheels and extension build tools, are pinned in `requirements.txt`; system drivers and the CUDA toolkit are installed separately.

```bash
git clone https://github.com/linxizhi/MIKU.git
cd MIKU
conda create -n miku python=3.10 -y
conda activate miku
python -m pip install -r requirements.txt
bash scripts/build_miku_extensions.sh
```

The build produces the custom CUDA rasterizer and C++ UV inpainting module inside the checkout. It detects the visible GPU architecture. For a build machine without a visible GPU, set `TORCH_CUDA_ARCH_LIST` for the target hardware. Set `CUDA_HOME` if the toolkit is not at `/usr/local/cuda`, and `MAX_JOBS` to control compiler parallelism.

## Model weights

**This commit contains code only.** MIKU albedo and PBR adapter weights are not included and their download links are pending. Place the downloaded assets in the following layout before running inference:

```text
checkpoints/
├── flux2-klein-base-9b/              # complete FLUX.2 Klein base-9B Diffusers snapshot
│   ├── model_index.json
│   ├── scheduler/
│   ├── tokenizer/
│   ├── text_encoder/
│   ├── transformer/
│   └── vae/
├── albedo/
│   └── pytorch_lora_weights.safetensors
├── pbr/
│   ├── pytorch_lora_weights.safetensors
│   └── dino_linear.pt
└── dinov2/
    └── dinov2_vitl14_reg4_pretrain.pth
```

Use the **base 9B** checkpoint, not a distilled or different-size model. The DINO backbone is **ViT-L/14 with four registers**. The inference-only backbone code is included, so neither a DINO training checkout nor a hidden Torch Hub cache is required. Base-model and DINO weights retain their upstream licenses.

By default, paths are resolved under this repository's `checkpoints/` directory. Use `--weights-dir /path/to/checkpoints` or `MIKU_CHECKPOINT_DIR` to select another root. Individual relative paths in `cfgs/miku.yaml` are resolved against that root; absolute path overrides remain supported. Missing files produce an explicit error before model loading. Inference reads local weights and does not download them implicitly.

```bash
python tools/check_weights.py
# Alternative location:
python tools/check_weights.py --weights-dir /path/to/checkpoints
```

## Inference

Supply a mesh and one reference image. The mesh is normalized for rendering and exported back in its original coordinates. UVs are unwrapped by default; the same six cameras are used for conditioning, prediction and baking.

```bash
bash scripts/run_miku.sh \
  --mesh inputs/mesh.glb \
  --image inputs/reference.png \
  --output outputs/example/textured_mesh.obj
```

This runs **albedo → MR → UV baking → vertex/OpenCV inpainting → OBJ and GLB export**. The GLB embeds lossless albedo and metallic–roughness textures. In glTF, **G = roughness** and **B = metallic**.

| Setting | Default | CLI override |
| --- | --- | --- |
| Geometry render / restored camera view | 1024 × 1024 | `--render-size 1024 --resolution 1024` |
| UV texture | 2048 × 2048 | `--texture-size 2048` |
| Albedo steps / seed / geometry CFG | 10 / 0 / 2.0 | `--steps 10 --seed 0 --guidance 2` |
| Albedo packed canvas longest side | 2048 | `miku_max_canvas_size` in YAML |
| PBR steps / seed | 20 / 99 | `--pbr-steps 20 --pbr-seed 99` |
| PBR layout | `pack` | `--pbr-mode pack` or `--pbr-mode nonpack` |
| MR background whitening | enabled | `--no-pbr-background-white` to disable |
| Model offload | model CPU offload | `--offload model` for albedo; `pbr_offload` in YAML |

**Pack:** resize the generated albedo atlas to half its width and height, then pad the right/bottom edge to a multiple of 16. **Nonpack:** restore the six full camera frames, resize each to 512 × 512 and arrange a 3 × 2 **1536 × 1024** grid. Both layouts preserve view order and restore the original camera frames before baking.

The renderer's geometry masks are passed to MR postprocessing. Background pixels become white after prediction; foreground values are preserved. Raw float and PNG predictions are kept separately in the debug directory. Geometry CFG for albedo preserves the reference image and masks out geometry latents in the unconditional branch.

```bash
# Use a different weight directory and nonpack MR.
bash scripts/run_miku.sh --weights-dir /path/to/checkpoints \
  --mesh inputs/mesh.glb --image inputs/reference.png \
  --pbr-mode nonpack --output outputs/nonpack/textured_mesh.obj

# Geometry preview without loading any model weights.
bash scripts/run_miku.sh --mesh inputs/mesh.glb \
  --conditions-only --output outputs/conditions/textured_mesh.obj

# Skip MR, or save predicted views without baking.
# Add --albedo-only or --mv-only to a normal inference command.
```

Outputs include `textured_mesh.glb`, OBJ/MTL, albedo PNG, roughness PNG, metallic PNG and a combined MR PNG. `textured_mesh_debug/` stores geometry masks, normal/position conditions, packing metadata, six-view predictions and inference reports.

## Repository structure

```text
MIKU/
├── infer_miku.py                    # command-line inference
├── mikuPipeline.py                  # albedo → MR orchestration
├── miku_config.py
├── cfgs/miku.yaml
├── miku/                           # model forward and strict checkpoint loading
│   └── vendor/dinov2/              # inference backbone subset + license
├── DifferentiableRenderer/         # rendering, UV baking and inpainting
├── custom_rasterizer/              # CUDA extension source
├── utils/                          # view packing, masking and UV utilities
├── scripts/                        # build and inference launchers
├── tools/                          # checkpoint and exported-result validation
├── tests/                          # inference regression checks
├── requirements.txt
└── docs/                           # project page and paper assets
```

The release contains no training entry point, dataset pipeline, optimizer or training-resume utilities. Checkpoints, generated outputs and compiled binaries are excluded from version control.

## Acknowledgments

MIKU builds upon prior work in texture and physically based material generation, including [LumiTex](https://github.com/LumiTexPBR/LumiTex), [UniTEX](https://github.com/YixunLiang/UniTEX), and [MaterialMVP](https://github.com/ZebinHe/MaterialMVP). We thank their authors for the ideas and publicly shared resources that helped make this work possible.

Third-party source and license notes are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
