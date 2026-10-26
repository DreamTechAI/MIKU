# MIKU

### Material Invariance and Knowledge Unification for PBR Generation

**Yajie Bao**<sup>1,2</sup>, **Jiaye Fu**<sup>3,2</sup>, **Youtian Lin**<sup>1</sup>, **Shuang Wu**<sup>1,2</sup>, **Jiachen Qian**<sup>4,2</sup>, **Feihu Zhang**<sup>2</sup>, **Xun Cao**<sup>1</sup>, **Yao Yao**<sup>1</sup>

<sup>1</sup> Nanjing University · <sup>2</sup> DreamTech · <sup>3</sup> Peking University · <sup>4</sup> The University of Hong Kong

[**Project page source**](docs/index.html) · [**Paper PDF**](docs/paper.pdf) · [**Publication DOI**](https://doi.org/10.1145/3842521)

> **Congratulations 🎉 Accepted to SIGGRAPH Asia TOG!**

![A scene assembled from AI-generated meshes, with textures and PBR material maps synthesized by MIKU.](docs/assets/teaser.webp)

MIKU combines standardized material representations with two-stage multi-view generation to produce detailed albedo and coherent metallic–roughness maps for 3D assets.

## Overview

Material generation is complicated by inconsistent annotations, rendering conventions, and parameterizations across 3D asset collections. Similar appearances may correspond to substantially different material parameters, making supervision ambiguous.

MIKU addresses this problem through a unified PBR data processing pipeline and a two-stage generation framework. The data pipeline brings heterogeneous assets into a shared representation using semantic material priors, controlled illumination, material-aware filtering, and detail-preserving refinement. Geometry-conditioned multi-view generation then synthesizes illumination-invariant albedo, followed by deterministic albedo-conditioned metallic–roughness estimation. The resulting materials improve appearance fidelity and consistency across object parts and viewpoints under the paper's evaluation protocol.

## Method

1. **Unify the data.** Consolidate assets under a shared PBR specification with 20 semantic material categories and four optional metal subclasses. Material recognition and segmentation support region-level normalization while preserving fine details. Filtering and validation reduce the initial pool of approximately 260K assets to approximately 20K PBR-consistent training samples.
2. **Generate multi-view albedo.** Pack views efficiently and condition a diffusion model on a reference image and geometry. Geometry-conditioned classifier-free guidance strengthens structural consistency. Distribution matching distillation reduces the albedo sampling process from 30 steps to six steps.
3. **Estimate metallic and roughness.** Predict aligned MR maps from the generated albedo and a shaded reference image using deterministic inference within a diffusion framework. DINO and VAE features provide semantic and appearance conditioning.

The framework predicts albedo at **1024 × 1024** and metallic–roughness maps at **512 × 512**. The manuscript reports **40 + 12 seconds** for the two generation stages under its experimental setup.

## Results

The manuscript evaluates texture generation on **200 Objaverse assets**, rendering six random viewpoints per asset. The table below reproduces its main appearance-quality results; lower is better for FID, CMMD, LPIPS, and CLIP-FID, and higher is better for CLIP-I.

| Method | FID ↓ | CMMD ↓ | LPIPS ↓ | CLIP-I ↑ | CLIP-FID ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MV-Adapter | 165.49 | 2.2235 | 0.2383 | 0.9097 | 22.911 |
| UniTEX | 161.27 | 2.1898 | 0.2174 | 0.9052 | 22.623 |
| Hunyuan3D-2.1 | 154.88 | 2.2625 | 0.2167 | 0.9132 | 20.132 |
| LumiTex | 144.61 | 1.8677 | 0.1992 | 0.9215 | 16.182 |
| **MIKU** | **121.25** | **1.4652** | **0.1612** | **0.9411** | **13.358** |

## Code and Model Weights

Source code and pretrained model weights will be released in this repository.

## Acknowledgments

MIKU builds upon prior work in texture and physically based material generation, including [LumiTex](https://github.com/LumiTexPBR/LumiTex), [UniTEX](https://github.com/YixunLiang/UniTEX), and [MaterialMVP](https://github.com/ZebinHe/MaterialMVP). We thank their authors for the ideas and publicly shared resources that helped make this work possible.
