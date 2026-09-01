# MIKU

### Material Invariance and Knowledge Unification for PBR Generation

<p align="center">
  <a href="https://doi.org/10.1145/3842521">
    <img src="https://img.shields.io/badge/Paper-ACM%20TOG-blue" alt="Paper">
  </a>
  <a href="https://github.com/YOUR_USERNAME/MIKU">
    <img src="https://img.shields.io/github/stars/YOUR_USERNAME/MIKU?style=social" alt="GitHub stars">
  </a>
</p>

Official implementation of:

> **MIKU: Material Invariance and Knowledge Unification for PBR Generation**

MIKU is a unified framework for physically consistent PBR material generation. It standardizes geometry and texture representations across heterogeneous datasets and uses a two-stage geometry-aware diffusion framework to generate consistent albedo, metallic, and roughness maps across multiple views and semantic object parts.

## Overview

Existing PBR generation pipelines often use different material definitions, rendering conventions, channel ranges, and parameterization schemes. These inconsistencies make cross-dataset training difficult and may lead to unstable or physically implausible material predictions.

MIKU addresses these issues with:

- A unified PBR material representation;
- A standardized geometry and texture processing pipeline;
- Material parameter normalization across different datasets;
- Geometry-aware multi-view albedo generation;
- Metallic–roughness generation conditioned on albedo and geometry;
- Improved consistency across views and semantic object parts.

## Method

MIKU contains two main stages:

1. **Geometry-aware albedo generation**

   Synthesizes multi-view albedo maps with explicit geometric guidance to improve appearance consistency across views.

2. **Metallic–roughness generation**

   Predicts physically plausible metallic and roughness maps conditioned on the generated albedo and unified material representation.

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="MIKU overview">
</p>

## Qualitative Results

<p align="center">
  <img src="assets/qualitative_comparison.png" width="95%" alt="Qualitative comparison">
</p>

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MIKU.git
cd MIKU

conda create -n miku python=3.10
conda activate miku

pip install -r requirements.txt
