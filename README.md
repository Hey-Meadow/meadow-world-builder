# Meadow World Builder: Single-Image 3D Gaussian Splatting on Apple Silicon

An [MLX](https://github.com/ml-explore/mlx) re-implementation of single-image → 3D Gaussian Splatting reconstruction, runnable end-to-end on a single Apple Silicon Mac. No CUDA, no PyTorch at inference, no cloud GPU.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![MLX](https://img.shields.io/badge/MLX-0.21+-orange.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3%2FM4-black.svg)](https://www.apple.com/mac/)
[![M1 Max](https://img.shields.io/badge/M1_Max-~100_s%2Fobj_(18×)-success.svg)](docs/FINAL_BENCHMARK.md)

<table>
  <tr>
    <td align="center"><img src="assets/gallery/chair_clean.gif" width="180"/><br/><sub>chair</sub></td>
    <td align="center"><img src="assets/gallery/table_clean.gif" width="180"/><br/><sub>table</sub></td>
    <td align="center"><img src="assets/gallery/objectf_clean.gif" width="180"/><br/><sub>misc-1</sub></td>
    <td align="center"><img src="assets/gallery/plush.gif" width="180"/><br/><sub>plush</sub></td>
    <td align="center"><img src="assets/gallery/objecte.gif" width="180"/><br/><sub>misc-2</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/gallery/toy_1.gif" width="180"/><br/><sub>toy 1</sub></td>
    <td align="center"><img src="assets/gallery/toy_2.gif" width="180"/><br/><sub>toy 2</sub></td>
    <td align="center"><img src="assets/gallery/toy_3.gif" width="180"/><br/><sub>toy 3</sub></td>
    <td align="center"><img src="assets/gallery/toy_4.gif" width="180"/><br/><sub>toy 4</sub></td>
    <td align="center"><img src="assets/gallery/toy_5.gif" width="180"/><br/><sub>toy 5</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/gallery/toy_6.gif" width="180"/><br/><sub>toy 6</sub></td>
    <td align="center"><img src="assets/gallery/toy_7.gif" width="180"/><br/><sub>toy 7</sub></td>
    <td align="center"><img src="assets/gallery/toy_8.gif" width="180"/><br/><sub>toy 8</sub></td>
    <td align="center"><img src="assets/gallery/toy_9.gif" width="180"/><br/><sub>toy 9</sub></td>
    <td align="center"><img src="assets/gallery/toy_10.gif" width="180"/><br/><sub>toy 10</sub></td>
  </tr>
</table>

<sub>All 15 objects above were reconstructed from a single RGB image + mask on an Apple M1 Max, ~100 s end-to-end each. Frames are rendered via macOS Quick Look on the exported `.ply` files (also shipped in <a href="assets/demos">assets/demos</a>).</sub>

> **TL;DR.** A faithful MLX port of Meta's SAM 3D Objects inference stack — MoGe depth → Sparse-Structure DiT → SLAT DiT → Gaussian decoder — with shortcut distillation, bf16 mixed precision, custom Metal sparse-attention kernels, and `gs_4` decoder slimming. Brings single-image 3DGS reconstruction from "needs an A100" to "runs on your laptop while you keep working".

---

## Table of Contents

1. [Highlights](#highlights)
2. [Benchmark](#benchmark)
3. [Quality vs Reference Implementation](#quality-vs-reference-implementation)
4. [Installation](#installation)
5. [Pre-trained Checkpoints](#pre-trained-checkpoints)
6. [Quickstart](#quickstart)
7. [Optimization Stack](#optimization-stack)
8. [Pipeline](#pipeline)
9. [Limitations](#limitations)
10. [Roadmap](#roadmap)
11. [Relationship to SAM 3D Objects](#relationship-to-sam-3d-objects)
12. [Acknowledgements](#acknowledgements)
13. [Citation](#citation)

---

## Highlights

- **Single-Mac inference.** Full pipeline in pure MLX on Apple Silicon. No CUDA, no `torch` at inference, no Docker.
- **~18× faster than the unoptimized M1 baseline.** Chair 1800 s → 86 s. Table 1800 s → 94 s. Plush 1800 s → 122 s. Mean ~100 s / object on M1 Max.
- **Numerically validated.** Quaternion / scale / opacity distributions match the reference web-demo outputs within published tolerances ([`docs/FINAL_BENCHMARK.md`](docs/FINAL_BENCHMARK.md)).
- **Streaming-friendly output.** Native `.ply` for SuperSplat and any standard 3DGS viewer, plus `.spz` (~7 MB) for web delivery.
- **Reproducible.** One CLI entry point (`meadow3d/infer.py`), pinned weight-conversion script, ablation flags exposed end-to-end.

## Benchmark

End-to-end wall-clock from `meadow3d/infer.py` on an Apple **M1 Max** (10-core CPU, 32-core GPU, 64 GB unified memory), Python 3.11.12, MLX 0.21:

| Object | Wall total | MoGe | SS DiT (4-step shortcut) | SLAT DiT (25-step CFG-5) | GS decode | Prune | Output |
|---|---:|---:|---:|---:|---:|---:|---:|
| chair |  **86 s** | 1.53 s |  7.76 s |  71.77 s | 0.78 s | 0.05 s | 63 624 Gaussians |
| table |  **94 s** | 1.56 s |  8.72 s |  79.00 s | 0.83 s | 0.07 s | 64 000 Gaussians |
| plush | **122 s** | 1.64 s | 10.25 s | 104.81 s | 1.08 s | 0.05 s | 64 000 Gaussians |

Mean **~100 s / object**. SLAT diffusion (25 CFG steps × 2 forward passes) accounts for **80–86 %** of wall time and is the next obvious optimization target — a SLAT shortcut at the SS shortcut's quality would project end-to-end to ~25–35 s.

### Speedup vs unoptimized M1 Max baseline

| Object | Baseline (no shortcut, fp32, no Metal kernel) | This port | Speedup |
|---|---:|---:|---:|
| chair | 1800 s | 86 s  | **20.9×** |
| table | 1800 s | 94 s  | **19.1×** |
| plush | 1800 s | 122 s | **14.7×** |

## Quality vs Reference Implementation

We compared Meadow World Builder outputs against PLYs downloaded directly from the [official Meta SAM 3D Objects web demo](https://aidemos.meta.com/segment3d) on identical inputs.

| Metric | chair (Meadow World Builder vs ref) | table (Meadow World Builder vs ref) | plush (Meadow World Builder vs ref) |
|---|---|---|---|
| Gaussian count | 63 624 / 68 076 (−7 %) | 64 000 / 64 380 (−0.6 %) | 64 000 / 51 340 (+25 %) |
| Bounding box (x,y,z) | match within 12 % | match within 4 % | wider/looser cloud |
| Opacity mean / median | 0.943 / 0.981 vs 0.897 / 0.973 | 0.971 / 0.992 vs 0.976 / 0.991 | 0.866 / 0.933 vs 0.980 / 0.993 |
| Quaternion `\|q\|` | 1.0000 (both) | 1.0000 (both) | 1.0000 (both) |

- **Chair, table:** geometry and bounding box visually indistinguishable from the reference. Minor colour-cast on chair (slightly darker red).
- **Plush:** geometry correct; cloud fluffier (lower opacity, ~2× mean scale) — see [§5 of the benchmark report](docs/FINAL_BENCHMARK.md#5-remaining-gaps-honest).

## Installation

Requirements: **macOS 13.5+**, Apple Silicon (M1 / M2 / M3 / M4), **Python 3.11**, **24 GB+ unified memory** recommended.

```bash
git clone https://github.com/Hey-Meadow/meadow-world-builder
cd meadow3d

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> **Note.** Use `python3.11` explicitly — Python 3.14 on Apple Silicon currently segfaults during MLX graph compilation.

## Pre-trained Checkpoints

Weights are **not bundled with this repo** — they must be downloaded from Meta's official release and one-time converted to MLX format. See [`docs/CHECKPOINT_REPORT.md`](docs/CHECKPOINT_REPORT.md) for the exact source URLs and conversion procedure.

```bash
python meadow3d/scripts/convert_weights.py \
    --pt-dir path/to/meta_release/ \
    --out-dir meadow3d/weights/
```

This produces:

| File | Source module | Size |
|---|---|---:|
| `ss_dit_mlx.npz` | sparse-structure DiT | 1.1 GB |
| `slat_dit_mlx.npz` | SLAT DiT | 2.4 GB |
| `gs_decoder_gs4_mlx.npz` | Gaussian decoder (4 splats / voxel) | 180 MB |
| `moge_vitl_mlx.npz` | MoGe ViT-L depth backbone | 1.3 GB |

Total ≈ **5.0 GB** on disk. Weights inherit the licence of their respective upstream sources — see [Relationship to SAM 3D Objects](#relationship-to-sam-3d-objects) below.

## Quickstart

Single image + mask in, `.ply` out:

```bash
python meadow3d/infer.py \
    --image path/to/image.png \
    --mask  path/to/mask.png \
    --use-moge --use-shortcut --dtype mixed --prune-outliers \
    --out outputs/my_object.ply
```

Or supply a pre-merged RGBA image:

```bash
python meadow3d/infer.py --rgba combined.png --out outputs/my_object.ply
```

Export web-ready compressed splat (`.spz`, ~7 MB):

```bash
python meadow3d/infer.py \
    --image image.png --mask mask.png \
    --format both --out outputs/my_object.ply
# writes my_object.ply (4.3 MB) AND my_object.spz (~7 MB)
```

Render a 360° turntable GIF of any `.ply`:

```bash
bash meadow3d/scripts/ql_gif_pipeline.sh outputs/my_object.ply preview.gif 36 320
```

## Optimization Stack

Every flag is independent and ablation-friendly:

| Optimization | Flag / env | Effect |
|---|---|---|
| `gs_4` decoder swap | `SLAT_GS_VARIANT=gs_4` (default) | 4 splats / voxel; caps PLY at ~64 k Gaussians, removes 4 of 8 decode heads |
| Quaternion / scale fixes | always on | `qn` unit-normalize, log-scale clamp at 9e-4 (σ ≤ 0.010) — kills "stretchy" outliers |
| Outlier prune | `--prune-outliers` | radius-graph KNN prune as safety net for noisy MoGe outputs |
| SS shortcut model | `--use-shortcut` | SS sampler: 25-step CFG-7 → **4-step distilled**, ~6.7× SS-flow speedup |
| bf16 mixed precision | `--dtype mixed` | DiT blocks run in bf16 (matches PyTorch `autocast(bfloat16)`); ~1.4× DiT speedup |
| MoGe in MLX | `--use-moge` | depth via MLX port of MoGe ViT-L, ~1.5 s |
| Sparse Metal kernel | always on | hand-rolled Metal sparse attention for SLAT DiT blocks |

See [`docs/MATH_OPTIMIZATION_OPPORTUNITIES.md`](docs/MATH_OPTIMIZATION_OPPORTUNITIES.md) for the remaining optimization backlog.

## Pipeline

```
RGB + mask
   │
   ▼   (~1.5 s)
MoGe ViT-L  ────►  depth map
   │
   ▼   (~10 s)
Sparse-Structure DiT  ────►  occupied voxel grid (≤16 000 voxels)
   │       (4-step shortcut)
   ▼   (~80 s)
SLAT DiT  ────►  per-voxel structured latent
   │       (25-step CFG-5)
   ▼   (~1 s)
Gaussian Decoder (gs_4)  ────►  4 Gaussians / voxel  (≤64 000)
   │
   ▼   (~0.05 s)
Outlier Prune
   │
   ▼
.ply  /  .spz
```

Each stage is independently importable from `meadow3d/models/` — see [`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) for the module map.

## Limitations

1. **SLAT diffusion is still the bottleneck.** No distilled SLAT shortcut yet; the SS shortcut alone leaves 80 %+ of wall time on the SLAT stage.
2. **Hard scale clamp.** Splat scales are clamped at σ ≤ 0.010. The reference allows σ up to ~0.021; the trade is fewer outliers but slightly flatter fine detail (most visible on the plush face).
3. **Plush appearance gap.** Lower opacity mean, larger bbox, and ~2× mean scale vs the reference on this single class — likely under-fit SLAT features. Tracked in [`docs/PLUSH_EYES_FIX_REPORT.md`](docs/PLUSH_EYES_FIX_REPORT.md).
4. **`gs_4` only.** Reference uses `gs_8` (8 splats / voxel) plus decode-time pruning. We expose only the `gs_4` decoder for now; `gs_8` is on the roadmap.
5. **No video / multi-frame support.** Single-image inference only.

## Roadmap

- [ ] SLAT shortcut distillation (4-step SLAT → ~25–35 s end-to-end)
- [ ] `gs_8` decoder support + decode-time pruning
- [ ] Auto-mask via SAM 2 / SAM 3 in MLX (`--auto-mask` is stubbed)
- [ ] FlashAttention-style fused kernel for SLAT DiT
- [ ] M4 Pro / Max benchmark numbers
- [ ] WebGL viewer with native `.spz` streaming

## Relationship to SAM 3D Objects

**Meadow World Builder is an independent third-party port. It is not affiliated with, endorsed by, or maintained by Meta.**

- The **architecture** (sparse-structure DiT, SLAT DiT, Gaussian decoder, MoGe backbone) is described in Meta's [SAM 3D Objects](https://ai.meta.com/research/publications/sam-3d-objects/) paper and other prior work (TRELLIS, MoGe). Meadow World Builder is a from-scratch MLX re-implementation of that architecture.
- The **model weights** are Meta's. This repository contains *no* model weights — users must download the official Meta release themselves and run the provided conversion script. Redistribution of Meta's weights is governed by Meta's licence terms; please consult the upstream release before redistributing converted weights.
- The **MLX code, Metal kernels, port scripts, benchmarks, and documentation** in this repository are original work by the Meadow World Builder authors and are released under Apache 2.0.

If you publish results obtained with this port, please cite both Meta's original work and this port (see [Citation](#citation)).

## Acknowledgements

Built on top of:

- [SAM 3D Objects](https://ai.meta.com/research/publications/sam-3d-objects/) (Meta AI) — original architecture and weights.
- [MoGe](https://github.com/microsoft/moge) (Microsoft Research) — monocular geometry backbone.
- [TRELLIS](https://github.com/Microsoft/TRELLIS) (Microsoft Research) — sparse-structure / SLAT formulation.
- [MLX](https://github.com/ml-explore/mlx) (Apple) — array framework, autograd, Metal kernels.
- [SuperSplat](https://github.com/playcanvas/supersplat) — `.ply` viewer used for the gallery renders.

## Citation

If you use this port in your work, please cite both the original Meta paper and Meadow World Builder:

```bibtex
@article{meta_sam3d_objects_2025,
  title  = {SAM 3D Objects: Single-Image 3D Gaussian Splatting at Scale},
  author = {Meta AI Research},
  year   = {2025},
  note   = {https://ai.meta.com/research/publications/sam-3d-objects/}
}

@misc{huang_meadow_2026,
  title  = {Meadow World Builder: Single-Image 3D Gaussian Splatting on Apple Silicon},
  author = {Sheng-Kai Huang},
  year   = {2026},
  note   = {https://github.com/Hey-Meadow/meadow-world-builder}
}
```
