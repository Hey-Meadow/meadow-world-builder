# YoNoSplat MLX — first end-to-end benchmark

End-to-end forward pass on M1 Max, fp32, real re10k checkpoint.

## Setup

| Item | Value |
|---|---|
| Hardware | Apple M1 Max (32 GB unified memory) |
| MLX | 0.31.1 |
| Checkpoint | `re10k_224x224_ctx2to32.ckpt` (3.86 GB, 965 M params) |
| Input | `test_input.npz` — 1 batch × 2 views × 3 × 224 × 224 |
| Output | 100,352 Gaussians (1 × 2 × 50,176 × 1 surface) + 2 c2w poses |

## Timing (warm, 5 runs)

| Metric | ms |
|---|---:|
| Median | **363** |
| Mean | 363 |
| Min | 358 |
| Max | 366 |
| Per-run | 363, 361, 366, 358, 366 |

## Sanity values (from a single run)

| Field | Range / value |
|---|---|
| `camera_poses[0, 0]` | ≈ identity (4×4) — view-1-centric ✓ |
| `camera_poses[0, 1]` | near-identity rotation, ‖t‖ ≈ 0.013 |
| `intrinsic_pred` | fx ≈ 0.64, fy ≈ 0.60 (normalised, plausible for 224 inputs) |
| Gaussian scales | [0, 0.239] (capped at 0.3 by `UnifiedGaussianAdapter` ✓) |
| Gaussian opacities | [0, 0.914] (sigmoid output ✓) |
| `local_points` z | [0.755, 2.961] (positive depths via exp ✓) |

## Output shape contract (verified)

```
gaussians.means      : (1, 2, 50176, 1, 1, 3)
gaussians.scales     : (1, 2, 50176, 1, 1, 3)
gaussians.rotations  : (1, 2, 50176, 1, 1, 4)
gaussians.opacities  : (1, 2, 50176, 1, 1)
gaussians.harmonics  : (1, 2, 50176, 1, 1, 3, 1)
camera_poses         : (1, 2, 4, 4)
intrinsic_pred       : (1, 2, 2)
local_points         : (1, 2, 224, 224, 3)
```

## Caveats — what this number is NOT

This is the **end-to-end-runnable** milestone, not the **numerically-equivalent-to-PT** milestone. Four known stubs sit in the pipeline:

1. **`intrinsics_embed_layer`** not ported — patch tokens skip the small
   intrinsic-conditioning add. Zero-init at construction so for re10k the
   bias is small but non-zero.
2. **CroCo dual-layer concat** — upstream concats last-two decoder block
   outputs to a 2048-dim tensor. Agent B's port returns only the final
   block, and the assembler tiles `[x, x]` along the channel axis so the
   sub-decoders (`in_dim=2048`) accept the shape. **Architecturally wrong**
   for PT parity.
3. **Pos-embed bicubic interp** is done at load time on the host via
   PyTorch's `F.interpolate(mode='bicubic')`. Should match upstream exactly
   (same algorithm) but worth verifying.
4. **Heads `pixel_shuffle`** lives in the assembler, not the head module —
   functional but architecturally split across files.

Until those four are addressed, treat the 363 ms as a **lower bound** on
M1 Max throughput. The fixes don't add measurable cost (small adds /
extra concat / unchanged shape).

## Comparison to single-image port

| Model | Hardware | Wall time |
|---|---|---:|
| `meadow_wb` Hunyuan single-image (this repo, shipped) | M1 Max fp16 | ~25 s |
| YoNoSplat MLX (this benchmark) | M1 Max fp32, 2 view | **0.36 s** |
| Hunyuan single-image torch baseline | A100 fp16 | ~5 s |
| YoNoSplat torch baseline | A100 fp32, 2 view | (RunPod ref TBD) |

YoNoSplat is **~70× faster** than the single-image port on the same device. The architecture trades scene-scale-only output (no body / object mesh) for direct 3DGS prediction without iterative refinement — that's where the speedup comes from.

## Next benchmarks to run

- [ ] A100 reference run (RunPod, $1–3) — gives torch baseline + ground-truth render
- [ ] PSNR / SSIM vs torch reference once Tier-2 Metal rasterizer ships
- [ ] M3 Max / M4 timing once port is stable
- [ ] fp16 timing (currently fp32) — expected ≈ 1.7× speedup
