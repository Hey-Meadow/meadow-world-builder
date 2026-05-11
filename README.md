# Meadow - World Builder: Single-Image 3D Gaussian Splatting on Apple Silicon

An [MLX](https://github.com/ml-explore/mlx) re-implementation of single-image → 3D Gaussian Splatting reconstruction, runnable end-to-end on a single Apple Silicon Mac. No CUDA, no PyTorch at inference, no cloud GPU.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![MLX](https://img.shields.io/badge/MLX-0.21+-orange.svg)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3%2FM4-black.svg)](https://www.apple.com/mac/)
[![M1 Max](https://img.shields.io/badge/M1_Max-~31_s%2Fobj_(57×)-success.svg)](docs/FINAL_BENCHMARK.md)
[![Status](https://img.shields.io/badge/status-v0.0.2_alpha-yellow.svg)](#status)

<table>
  <tr>
    <td align="center"><img src="assets/gallery/chair_clean.gif" width="180"/><br/><sub>chair</sub></td>
    <td align="center"><img src="assets/gallery/table_clean.gif" width="180"/><br/><sub>table</sub></td>
    <td align="center"><img src="assets/gallery/oatchi.gif" width="180"/><br/><sub>Oatchi</sub></td>
    <td align="center"><img src="assets/gallery/moss.gif" width="180"/><br/><sub>Moss</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/gallery/pikmin_r.gif" width="180"/><br/><sub>red pikmin</sub></td>
    <td align="center"><img src="assets/gallery/pikmin_y.gif" width="180"/><br/><sub>yellow pikmin</sub></td>
    <td align="center"><img src="assets/gallery/pikmin_b.gif" width="180"/><br/><sub>blue pikmin</sub></td>
    <td align="center"><img src="assets/gallery/pikmin_p.gif" width="180"/><br/><sub>purple pikmin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/gallery/pikmin_light.gif" width="180"/><br/><sub>glow pikmin</sub></td>
    <td align="center"><img src="assets/gallery/toy_1.gif" width="180"/><br/><sub>toy 1</sub></td>
    <td align="center"><img src="assets/gallery/toy_2.gif" width="180"/><br/><sub>toy 2</sub></td>
    <td align="center"><img src="assets/gallery/toy_3.gif" width="180"/><br/><sub>toy 3</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/gallery/toy_4.gif" width="180"/><br/><sub>toy 4</sub></td>
    <td align="center"><img src="assets/gallery/toy_5.gif" width="180"/><br/><sub>toy 5</sub></td>
    <td align="center"><img src="assets/gallery/toy_6.gif" width="180"/><br/><sub>toy 6</sub></td>
    <td align="center"><img src="assets/gallery/toy_7.gif" width="180"/><br/><sub>toy 7</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/gallery/toy_8.gif" width="180"/><br/><sub>toy 8</sub></td>
    <td align="center"><img src="assets/gallery/toy_9.gif" width="180"/><br/><sub>toy 9</sub></td>
    <td align="center"><img src="assets/gallery/toy_10.gif" width="180"/><br/><sub>toy 10</sub></td>
    <td align="center"></td>
  </tr>
</table>

<sub>All 19 objects above were reconstructed from a single RGB image + mask on an Apple M1 Max, ~18–35 s end-to-end each (v0.0.2 default, curvature-cache on). Frames are rendered via macOS Quick Look on the exported `.ply` files (also shipped in <a href="assets/demos">assets/demos</a>).</sub>

<details>
<summary><b>Source images used for the character reconstructions above</b> (click to expand)</summary>

<br/>

<table>
  <tr>
    <td align="center"><img src="assets/sources/oatchi.png" height="140"/><br/><sub>Oatchi</sub></td>
    <td align="center"><img src="assets/sources/P4_Moss.png" height="140"/><br/><sub>Moss</sub></td>
    <td align="center"><img src="assets/sources/pikmin-r.png" height="140"/><br/><sub>red pikmin</sub></td>
    <td align="center"><img src="assets/sources/pikmin-y.png" height="140"/><br/><sub>yellow pikmin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/sources/pikmin-b.png" height="140"/><br/><sub>blue pikmin</sub></td>
    <td align="center"><img src="assets/sources/pikmin-p.png" height="140"/><br/><sub>purple pikmin</sub></td>
    <td align="center"><img src="assets/sources/pikmin-light.png" height="140"/><br/><sub>glow pikmin</sub></td>
    <td align="center"></td>
  </tr>
</table>

Each input is a single RGBA PNG with transparent background (the alpha channel doubles as the segmentation mask). All character likenesses © Nintendo, used here strictly for non-commercial demonstration of single-image 3D reconstruction; not for redistribution as standalone artwork.

</details>

> **TL;DR.** Server-grade single-image 3D Gaussian Splatting reconstruction in **~30 seconds** on Apple Silicon. MoGe depth → Sparse-Structure DiT → SLAT DiT → Gaussian decoder, end-to-end in pure MLX, with shortcut distillation, bf16 mixed precision, custom Metal sparse-attention kernels, curvature-cache temporal reuse, and decoder slimming. Brings 3DGS reconstruction from "needs an A100" to "runs on your laptop while you keep working".

---

## Table of Contents

1. [Highlights](#highlights)
2. [Benchmark](#benchmark)
3. [Output Quality](#output-quality)
4. [Installation](#installation)
5. [Pre-trained Checkpoints](#pre-trained-checkpoints)
6. [Quickstart](#quickstart)
7. [Troubleshooting: seed sensitivity](#troubleshooting-seed-sensitivity)
8. [Optimization Stack](#optimization-stack)
9. [Pipeline](#pipeline)
10. [Status](#status)
11. [Limitations](#limitations)
12. [Acknowledgements](#acknowledgements)
13. [Citation](#citation)

---

## Highlights

- **Single-Mac inference.** Full pipeline in pure MLX on Apple Silicon. No CUDA, no `torch` at inference, no Docker.
- **~57× faster than the unoptimized M1 baseline.** Chair 1800 s → 30 s. Table 1800 s → 32 s. Oatchi 1800 s → 32 s. Mean ~**31 s / object** on M1 Max.
- **Curvature-cache speedup** (Fast-SAM3D ②a) — on by default at `--slat-curvature-eps 0.5`, ~2.77× e2e vs no cache, quality-validated across chair / table / Oatchi.
- **Numerically validated.** Quaternion / scale / opacity distributions match the published-reference tolerances ([`docs/FINAL_BENCHMARK.md`](docs/FINAL_BENCHMARK.md)).
- **Streaming-friendly output.** Native `.ply` for SuperSplat and any standard 3DGS viewer, plus `.spz` (~7 MB) for web delivery.
- **Reproducible.** One CLI entry point (`meadow_wb/infer.py`), pretrained weights on HuggingFace, ablation flags exposed end-to-end.

## Benchmark

End-to-end wall-clock from `meadow_wb/infer.py` with v0.0.2 defaults (curvature cache on, eps=0.5) on an Apple **M1 Max** (10-core CPU, 32-core GPU, 64 GB unified memory), Python 3.11.12, MLX 0.21:

| Object | Wall total | MoGe | SS DiT | SLAT DiT (curv-cache) | GS decode | Prune | Output |
|---|---:|---:|---:|---:|---:|---:|---:|
| chair | **30.1 s** | 1.74 s |  7.72 s | 13.2 s (76% hits) | 0.79 s | 0.05 s | 63 624 Gaussians |
| table | **31.5 s** | 1.56 s |  8.72 s | 16.8 s (76% hits) | 0.83 s | 0.07 s | 64 000 Gaussians |
| Oatchi | **32.3 s** | 1.64 s | 10.25 s | 17.4 s (80% hits) | 1.08 s | 0.05 s | 64 000 Gaussians |

Mean **~31 s / object**. To opt out of the curvature cache (full 25-step SLAT loop), pass `--no-slat-curvature-cache`.

### Speedup vs M1 Max baselines

| Object | unoptimized (fp32, no Metal, no cache) | classic MLX (v0.0.1, no cache) | **v0.0.2 (default)** | unopt → v0.0.2 |
|---|---:|---:|---:|---:|
| chair | 1800 s | 78.1 s | 30.1 s | **59.8×** |
| table | 1800 s | 85.4 s | 31.5 s | **57.1×** |
| Oatchi | 1800 s | 97.9 s | 32.3 s | **55.7×** |

Full per-stage and ablation breakdown: [`docs/FINAL_BENCHMARK.md`](docs/FINAL_BENCHMARK.md) §6.

### How does this compare to a cloud A100?

We ran the original PyTorch pipeline on an **NVIDIA A100 80GB PCIe** (RunPod) to capture reference activations for our MLX port. Three objects were processed end-to-end with full forward-hook instrumentation (every intermediate tensor saved to disk for numerical validation):

| Object | A100 wall, instrumented<sup>†</sup> | A100 clean estimate<sup>‡</sup> | M1 Max v0.0.2 |
|---|---:|---:|---:|
| obj 1 |  94 s | ~70 s | — |
| obj 2 | 105 s | ~80 s | — |
| obj 3 |  38 s | ~20 s | — |
| chair |  — | — | **30.1 s** |
| table |  — | — | **31.5 s** |
| Oatchi |  — | — | **32.3 s** |

<sub>† Includes ~16 s model load + heavy disk I/O from saving every intermediate tensor (script: `mlx_port/debug/scripts/dump_pt_reference.py`).</sub><br/>
<sub>‡ Estimated by subtracting model-load and dump-overhead from the instrumented wall time. We have not yet rerun on A100 without instrumentation; the clean number could be lower.</sub>

**Bottom line.** v0.0.2 on M1 Max is in the **same order of magnitude as a cloud A100** for this pipeline, with no cloud cost, no cold start, no data egress, and no GPU contention. A future SLAT shortcut distillation (~5–6× projected speedup on the SLAT stage) would push end-to-end into single-digit seconds.

## Output Quality

Per-object statistics from the standard `.ply` output, compared against a published reference on identical inputs:

| Metric | chair | table | Oatchi |
|---|---|---|---|
| Gaussian count (ours / ref) | 63 624 / 68 076 (−7 %) | 64 000 / 64 380 (−0.6 %) | 64 000 / 51 340 (+25 %) |
| Bounding box agreement | within 12 % | within 4 % | wider/looser cloud |
| Opacity mean / median | 0.943 / 0.981 | 0.971 / 0.992 | 0.866 / 0.933 |
| Quaternion `\|q\|` | 1.0000 | 1.0000 | 1.0000 |

- **Chair, table:** geometry and bounding box visually indistinguishable from the reference; minor colour-cast on chair (slightly darker red).
- **Oatchi:** geometry correct; cloud fluffier (lower opacity, ~2× mean scale).

Full numerics including per-stage timings, optimization ablations, and quality regressions: [`docs/FINAL_BENCHMARK.md`](docs/FINAL_BENCHMARK.md).

## Installation

Requirements: **macOS 13.5+**, Apple Silicon (M1 / M2 / M3 / M4), **Python 3.11**, **24 GB+ unified memory** recommended.

```bash
git clone https://github.com/Hey-Meadow/meadow-world-builder
cd meadow-world-builder

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> **Note.** Use `python3.11` explicitly — Python 3.14 on Apple Silicon currently segfaults during MLX graph compilation.

## Pre-trained Checkpoints

Pre-converted MLX weights are hosted on HuggingFace — no manual conversion needed:

```bash
hf auth login   # one-time, paste your HF token (free account works)
hf download akaiii/meadow-world-builder-weights \
    --local-dir meadow_wb/weights/sam3d_objects
```

<sub>Older `huggingface-cli login` / `huggingface-cli download` syntax is deprecated in `huggingface_hub >= 1.0` — use `hf` as shown above.</sub>

Contents (downloaded into `meadow_wb/weights/sam3d_objects/`):

| File | Source module | Size |
|---|---|---:|
| `ss_flow.npz`           | sparse-structure DiT | 3.3 GB |
| `slat_flow.npz`         | SLAT DiT | 2.0 GB |
| `ss_embedder.npz`       | image conditioning (SS) | 2.2 GB |
| `slat_embedder.npz`     | image conditioning (SLAT) | 2.2 GB |
| `ss_decoder.npz`        | SS occupancy decoder | 166 MB |
| `slat_decoder_gs_4.npz` | Gaussian decoder (4 splats / voxel) | 192 MB |
| `slat_decoder_gs.npz`   | Gaussian decoder (8 splats / voxel) | 192 MB |
| `moge_vitl.npz`         | MoGe ViT-L depth backbone | 1.2 GB |

Total ≈ **11.5 GB** on disk. Weights remain subject to their upstream licence — see [`UPSTREAM_LICENSES/`](UPSTREAM_LICENSES).

## Quickstart

Prerequisite: weights downloaded per [Pre-trained Checkpoints](#pre-trained-checkpoints) above (`meadow_wb/weights/sam3d_objects/*.npz` populated).

A bundled sample (Pexels-licensed cat photo + auto-generated mask) is shipped at `assets/sample/` so you can verify the install in one command:

```bash
python meadow_wb/infer.py \
    --image assets/sample/cat.jpg \
    --mask  assets/sample/cat_mask.png \
    --use-moge --use-shortcut --dtype mixed --prune-outliers \
    --out outputs/cat.ply
```

Expected wall time on M1 Max (v0.0.2 defaults): ~40 s; output `.ply` opens directly in [SuperSplat](https://superspl.at).

For your own input — single image + mask in, `.ply` out:

```bash
python meadow_wb/infer.py \
    --image path/to/image.png \
    --mask  path/to/mask.png \
    --use-moge --use-shortcut --dtype mixed --prune-outliers \
    --out outputs/my_object.ply
```

Or supply a pre-merged RGBA image:

```bash
python meadow_wb/infer.py --rgba combined.png --out outputs/my_object.ply
```

Export web-ready compressed splat (`.spz`, ~7 MB):

```bash
python meadow_wb/infer.py \
    --image image.png --mask mask.png \
    --format both --out outputs/my_object.ply
# writes my_object.ply (4.3 MB) AND my_object.spz (~7 MB)
```

Render a 360° turntable GIF of any `.ply`:

```bash
bash meadow_wb/scripts/ql_gif_pipeline.sh outputs/my_object.ply preview.gif 36 320
```

## Troubleshooting: seed sensitivity

The flow-matching sampler is **seed-sensitive** on ambiguous or low-detail inputs — flat-coloured, thin, or partially-occluded objects sometimes collapse into a featureless blob on the default `--seed 42`. The fix is cheap: retry with a different `--seed`.

<table>
  <tr>
    <td align="center"><img src="assets/gallery/pikmin_y_seed42_fail_v2.gif" width="220"/><br/><sub><code>--seed 42</code> (default) — featureless blob</sub></td>
    <td align="center"><img src="assets/gallery/pikmin_y.gif" width="220"/><br/><sub><code>--seed 123</code> — full character recovered</sub></td>
  </tr>
</table>

Same input image, same flags, only `--seed` changed. Yellow pikmin went from a melted yellow shape to a full character with leaf, eyes, and limbs.

**Recipe** when an output looks wrong:

```bash
for s in 1 7 42 123 2024; do
  python meadow_wb/infer.py --rgba YOUR.png --seed $s --out /tmp/try_$s.ply
done
# open each /tmp/try_*.ply in SuperSplat, keep the best
```

The cost is **~30 s per retry on M1 Max** — practical to try 3-5 seeds before falling back to other knobs (`--no-shortcut` for slower-but-stricter SS, `--slat-cfg 7` for higher conditioning).

## Optimization Stack

Every flag is independent and ablation-friendly:

| Optimization | Flag / env | Effect |
|---|---|---|
| `gs_4` decoder swap | `SLAT_GS_VARIANT=gs_4` (default) | 4 splats / voxel; caps PLY at ~64 k Gaussians, removes 4 of 8 decode heads |
| Quaternion / scale fixes | always on | `qn` unit-normalize, log-scale clamp at 9e-4 (σ ≤ 0.010) — kills "stretchy" outliers |
| Outlier prune | `--prune-outliers` | radius-graph KNN prune as safety net for noisy depth |
| SS shortcut model | `--use-shortcut` | SS sampler: 25-step CFG-7 → **4-step distilled**, ~6.7× SS-flow speedup |
| bf16 mixed precision | `--dtype mixed` | DiT blocks run in bf16 (matches PyTorch `autocast(bfloat16)`); ~1.4× DiT speedup |
| MoGe in MLX | `--use-moge` | depth via MLX port of MoGe ViT-L, ~1.5 s |
| Sparse Metal kernel | always on | hand-rolled Metal sparse attention for SLAT DiT blocks |
| Curvature cache (Fast-SAM3D ②a) | `--slat-curvature-cache` | tangent-reuse on SLAT 25-step loop; ~2.8× end-to-end on M1 Max |

See [`docs/MATH_OPTIMIZATION_OPPORTUNITIES.md`](docs/MATH_OPTIMIZATION_OPPORTUNITIES.md) for the remaining optimization backlog.

## Pipeline

```
RGB + mask
   │
   ▼   (~1.6 s)
MoGe ViT-L  ────►  depth map
   │
   ▼   (~8 s)
Sparse-Structure DiT  ────►  occupied voxel grid (≤16 000 voxels)
   │       (4-step shortcut)
   ▼   (~15 s)
SLAT DiT  ────►  per-voxel structured latent
   │       (25-step CFG-5, ~76% curvature-cache hits)
   ▼   (~1 s)
Gaussian Decoder (gs_4)  ────►  4 Gaussians / voxel  (≤64 000)
   │
   ▼   (~0.05 s)
Outlier Prune
   │
   ▼
.ply  /  .spz   (total ~31 s on M1 Max)
```

Each stage is independently importable from `meadow_wb/models/` — see [`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) for the module map.

## Status

**v0.0.2 (alpha, May 2026)** — second public release; curvature cache enabled by default.

| Component | State | Notes |
|---|---|---|
| Inference pipeline (MoGe + SS DiT + SLAT DiT + GS decoder) | ✅ verified | quality-validated on chair / table / Oatchi + 7 additional kidsroom-scene objects |
| Pretrained MLX weights on HuggingFace | ✅ shipped | one `hf download` away, ~11.5 GB total |
| Weight-conversion script (`convert_weights.py`) | ✅ available | only needed if you re-convert from upstream PyTorch checkpoints |
| Custom Metal sparse attention kernel | ✅ verified | bundled, used by SLAT DiT |
| `--use-shortcut` (4-step SS sampler) | ✅ verified | default-on |
| Curvature cache (Fast-SAM3D ②a) | ✅ shipped | default-on at `eps=0.5`, ~2.77× end-to-end vs no cache |
| `gs_8` decoder weights | ✅ available | shipped in HF release as `slat_decoder_gs.npz`; default code path uses `gs_4` |

End-to-end smoke test passing on M1 Max with mean **~31 s / object** (chair 30.1 s / table 31.5 s / Oatchi 32.3 s).

## Limitations

1. **SLAT diffusion is still the bottleneck.** No distilled SLAT shortcut yet; the SS shortcut alone leaves 80 %+ of wall time on the SLAT stage.
2. **Hard scale clamp.** Splat scales are clamped at σ ≤ 0.010. The reference allows σ up to ~0.021; the trade is fewer outliers but slightly flatter fine detail (most visible on the Oatchi face).
3. **Oatchi appearance gap.** Lower opacity mean, larger bbox, and ~2× mean scale vs the reference on this single class — likely under-fit SLAT features. Tracked in [`docs/PLUSH_EYES_FIX_REPORT.md`](docs/PLUSH_EYES_FIX_REPORT.md).
4. **`gs_4` is the default decode path.** `gs_8` weights are shipped in the HF release (`slat_decoder_gs.npz`) but the default `infer.py` flow uses `gs_4` (4 splats / voxel; ~64 k Gaussian cap). Switching to `gs_8` programmatically is wired but not yet exposed as a CLI flag.
5. **No video / multi-frame support.** Single-image inference only.

## Acknowledgements

This port incorporates and re-implements components from upstream model-architecture work, monocular geometry estimation, structured-latent diffusion, and the [MLX](https://github.com/ml-explore/mlx) array framework. Detailed upstream attributions and licences are bundled under [`UPSTREAM_LICENSES/`](UPSTREAM_LICENSES). `.ply` previews use [SuperSplat](https://github.com/playcanvas/supersplat).

## Citation

```bibtex
@misc{huang_meadow_2026,
  title  = {Meadow - World Builder: Single-Image 3D Gaussian Splatting on Apple Silicon},
  author = {Sheng-Kai Huang},
  year   = {2026},
  note   = {https://github.com/Hey-Meadow/meadow-world-builder}
}
```
