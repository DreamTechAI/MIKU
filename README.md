# MIKU

### Material Invariance and Knowledge Unification for PBR Generation

**Yajie Bao**<sup>1,2</sup>, **Jiaye Fu**<sup>3,2</sup>, **Youtian Lin**<sup>1</sup>, **Shuang Wu**<sup>1,2</sup>, **Jiachen Qian**<sup>4,2</sup>, **Feihu Zhang**<sup>2</sup>, **Xun Cao**<sup>1</sup>, **Yao Yao**<sup>1</sup>

<sup>1</sup> Nanjing University · <sup>2</sup> DreamTech · <sup>3</sup> Peking University · <sup>4</sup> The University of Hong Kong

Yajie Bao conducted this work during an internship at DreamTech. Yao Yao is the corresponding author.

[**Project page source**](docs/index.html) · [**Paper PDF**](docs/paper.pdf) · [**Publication DOI**](https://doi.org/10.1145/3842521) · [**BibTeX**](references.bib)

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

These metrics assess rendered appearance and semantic alignment. The work targets relative physical consistency under a unified specification; its artist-derived material priors are not measurements of real-world reflectance. Final dataset screening also involves human judgment. See the paper for the complete evaluation and limitations.

## Publication status

The repository is private. **Public GitHub Pages publishing is disabled** while the project page is being prepared. Use the local preview below to view the page privately. Enabling GitHub Pages from a private repository can still expose the website publicly; do not enable publication until the project is ready to be public.

## Repository contents

This repository currently hosts the **research project page**, paper, and supporting figures. Model inference code, training code, and checkpoints are not included in this release.

```text
docs/
  index.html       Project page
  assets/          Website images and supporting assets
  paper.pdf        Research paper
README.md          Project overview and acknowledgments
references.bib     MIKU and foundational-work citations
```

### Preview locally

From the repository root, run:

```bash
python3 -m http.server 8000 --directory docs
```

Open [http://localhost:8000](http://localhost:8000). The site is static and does not require a package installation or build step.

### Future public deployment with GitHub Pages

When public release is approved, the planned project-page address is [https://linxizhi.github.io/MIKU/](https://linxizhi.github.io/MIKU/). It is currently unpublished.

In the repository's **Settings → Pages**, select **Deploy from a branch**, choose the **main** branch and **/docs** folder, then save. Subsequent changes pushed to `main` are published from `docs/` by GitHub Pages.

## Acknowledgments

**MIKU is built upon prior research in texture and physically based material generation, including LumiTex, UniTEX, and MaterialMVP.** We gratefully acknowledge the authors of these works and the broader research community for the ideas and publicly shared resources that provided a foundation for this project. Please also cite these foundational works when relevant.

| Work | Paper | Project / code |
| --- | --- | --- |
| **LumiTex**: Towards High-Fidelity PBR Texture Generation with Illumination Context | [arXiv](https://arxiv.org/abs/2511.19437) | [Project](https://lumitex.vercel.app/) · [Code](https://github.com/LumiTexPBR/LumiTex) |
| **UniTEX**: Universal High Fidelity Generative Texturing for 3D Shapes | [arXiv](https://arxiv.org/abs/2505.23253) | [Code](https://github.com/YixunLiang/UniTEX) |
| **MaterialMVP**: Illumination-Invariant Material Generation via Multi-view PBR Diffusion | [arXiv](https://arxiv.org/abs/2503.10289) · [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/He_MaterialMVP_Illumination-Invariant_Material_Generation_via_Multi-view_PBR_Diffusion_ICCV_2025_paper.html) | [Project](https://zebinhe.github.io/MaterialMVP/) · [Code](https://github.com/ZebinHe/MaterialMVP) |

**中文致谢：** MIKU 是在 [LumiTex](https://github.com/LumiTexPBR/LumiTex)、[UniTEX](https://github.com/YixunLiang/UniTEX)、[MaterialMVP](https://github.com/ZebinHe/MaterialMVP) 等纹理与物理材质生成工作的基础上完成的。感谢这些工作的作者及相关研究社区所提出的方法和公开分享的资源，为本项目奠定了基础。使用或引用相关成果时，也请关注并引用这些基础工作。

This work was supported by the National Natural Science Foundation of China (62472213), the Gusu Innovation & Entrepreneurship Leading Talents Program (ZXL2024361), and computing resources from DreamTech.

## Citation

Publication metadata below follows the supplied manuscript: *ACM Transactions on Graphics*, volume 45, issue 6, article 181, December 2026.

```bibtex
@article{bao2026miku,
  title = {{MIKU}: Material Invariance and Knowledge Unification for {PBR} Generation},
  author = {Bao, Yajie and Fu, Jiaye and Lin, Youtian and Wu, Shuang and Qian, Jiachen and Zhang, Feihu and Cao, Xun and Yao, Yao},
  journal = {ACM Transactions on Graphics},
  volume = {45},
  number = {6},
  articleno = {181},
  year = {2026},
  month = dec,
  doi = {10.1145/3842521},
  url = {https://doi.org/10.1145/3842521}
}
```

Citation entries for MIKU, LumiTex, UniTEX, and MaterialMVP are available in [references.bib](references.bib).
