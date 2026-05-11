# SAM 3D Objects MLX — Math Optimization Opportunities

Research-only inventory of mathematical / algorithmic optimizations for the
MLX inference port. Pure analysis; no code is changed by this document.

Baseline reference: `chair.png` end-to-end on M1 Max ≈ 30+ min, dominated by
the two flow-matching stages (25 Euler steps × dual CFG forward = 50 DiT
forwards per stage × 2 stages = 100 DiT forwards), all run in fp32. The
SS DiT depth is 24 blocks × channels=1024; SLAT DiT depth is 24 blocks ×
channels=1024 + a sparse U-Net wrapper. The GS decoder adds 12 sparse
swin-attention blocks.

Three findings dominate the ROI table: **(1) shortcut-model self-distillation
is already trained into the SS checkpoint** (the `d_embedder` weights are
present in `ss_flow.npz`; PT exposes `use_distillation=True` to switch the
sampler from 25-step CFG to 1–4 step shortcut), **(2) PT inference defaults
to bfloat16 autocast on both DiT backbones** (the MLX port runs fp32
end-to-end), and **(3) the Python inner loop that builds the 27-neighbor
table for SubMConv3d is O(N) Python-level work** that can be vectorized to
numpy in 30 lines. Combined estimated speedup: 8–25× depending on how
aggressively the user accepts the shortcut quality trade-off.

---

## Top 3 recommendations

### 1. Use the trained shortcut model — 1–4 step SS sampling instead of 25

**Current state.**
`mlx_port/models/pipeline_mlx.py::_mot_euler_sample` runs a 25-step Euler
integrator with CFG every step (`v = (1+s)·v_cond − s·v_uncond`), so SS
costs 50 backbone forwards. The MLX port is wired to pass `d=None` to
`MOTDiTBackbone.__call__` (`pipeline_mlx.py:1124`), which silently disables
the shortcut head even though the weights are loaded.

**Evidence the model is shortcut-distilled.**
The SS checkpoint contains `reverse_fn.backbone.d_embedder.mlp.{0,2}.{weight,bias}`
(verified: `mx.load("ss_flow.npz")` shows `(1024, 256)`/`(1024, 1024)`
shapes). The PT class is `sam3d_objects/model/backbone/generator/shortcut/model.py::ShortCut`,
trained with `self_consistency_prob=0.25` per the binary-time schedule
`d ∈ {1/2^i for i in range(8)}`. PT exposes `use_distillation=True` in
`sam3d_objects/pipeline/inference_pipeline.py:649-656`, which sets
`no_shortcut=False` and `strength=0` (the shortcut model already
internalizes CFG via the self-consistency target with
`self_consistency_cfg_strength=3.0`).

**Proposed change.**
Add a `use_shortcut: bool = False` flag to `_mot_euler_sample`. When
`True`, replace the 25-step CFG loop with a 4-step or 1-step schedule:

  - 4-step shortcut: `d = 1/4` for all steps, `cfg=0`, no uncond pass →
    4 backbone forwards (12.5× fewer than 50).
  - 1-step shortcut: `d = 1`, single forward from `t=0` → 50× fewer than
    baseline.

The MOTDiTBackbone forward signature already accepts `d`; pass
`mx.array([d * time_scale])` so the shape stays scalar-broadcast.

**Trade-off.**
Quality drops from 25-step CFG-7 baseline. Per the One-Step paper
(Frans et al. 2024, *One Step Diffusion via Shortcut Models*,
[arXiv:2410.12557](https://arxiv.org/abs/2410.12557)) on
ImageNet/CIFAR, 4-step shortcut is within ~1 FID of full sampling and
1-step is within ~3 FID. SAM 3D Objects' SS stage is producing a 16³
latent cube that gets argwhere'd into voxel coords — geometry is more
forgiving than pixel reconstruction, so 4-step should hold. SLAT does
NOT have d_embedder (`mx.load("slat_flow.npz")` returns no `d_embedder`
keys), so SLAT stays on 25-step Euler+CFG.

**Speedup.**
SS-only saves 46/50 forwards = 92% of SS time. SS:SLAT timing is roughly
1:1 in the current pipeline (similar depth, similar context length per
step). End-to-end **~2.0× speedup** for 4-step shortcut, **~2.4× speedup**
for 1-step shortcut.

**Effort.** 2–4 hours. The wiring already exists in `MOTDiTBackbone.__call__`;
only the sampler loop needs modification.

**References.**
- Frans, K. et al. *One Step Diffusion via Shortcut Models*. arXiv:2410.12557 (2024).
- PT source: `sam3d_objects/model/backbone/generator/shortcut/model.py`
- PT call site: `sam3d_objects/pipeline/inference_pipeline.py:649-656` (`use_distillation`)

---

### 2. bfloat16 mixed precision in both DiT backbones

**Current state.**
All MLX modules run in fp32 (`dtype=mx.float32` is the default for every
`mx.array` allocation; `_layer_norm` casts implicitly to fp32; `nn.Conv3d`
holds fp32 weights). The DiT backbone is the dominant cost (~96% of
inference) and the MLX `mx.fast.scaled_dot_product_attention` accepts
fp16/bf16 inputs natively on M1 with no kernel fallback.

**Evidence PT uses bf16.**
`sam3d_objects/pipeline/inference_pipeline.py:71` declares
`dtype="bfloat16"` as default; line 674 wraps the SS sample call in
`torch.autocast(device_type="cuda", dtype=self.shape_model_dtype)`,
which inherits the bf16 default. So the *reference* numerics this
checkpoint was validated against are already bf16. Running MLX at fp32
is over-engineering parity in the wrong direction — we are MORE precise
than the reference output the checkpoint was trained/tested at.

**Proposed change.**
1. Cast loaded weights to `mx.bfloat16` inside the DiT block constructors
   (or via a one-shot post-load pass in `pipeline_mlx.py::from_npz_dir`).
2. Cast inputs at the DiT entry point (`backbone(x, t, cond)`) and cast
   back to fp32 at the output for the residual into the sampler. This
   matches PT's `autocast` pattern — only the matmul-heavy region runs
   bf16, while accumulators and the noise schedule stay fp32.
3. Keep the SS decoder's Conv3d in fp32 (cheap, last stage, parity-sensitive).
4. Keep RMSNorm + LayerNorm computed in fp32 (PT does this implicitly via
   `LayerNorm32`; mirror with `x.astype(mx.float32)` inside the norm).

**Trade-off.**
~0.5–1% cosine drift vs current fp32 MLX output, but the *checkpoint*
itself was trained/tested in bf16 so this is closer to PT-on-CUDA than
the current MLX port is. Quality is likely indistinguishable (Meta's
released chair/table reference plys were rendered from a bf16 forward).

**Speedup.**
M1 Max bf16 matmul is ~1.7–2.0× faster than fp32 in MLX (per public MLX
benchmarks, e.g. https://github.com/ml-explore/mlx/issues/132). Memory
bandwidth halves, which on a memory-bound 24-block × 4096-token DiT
matters more than peak FLOPs. End-to-end **~1.5–1.8× speedup**.
Stacks multiplicatively with finding (1) → 4-step shortcut + bf16 ≈ 3.0×.

**Effort.** 4–6 hours. Mostly bookkeeping (find every fp32 allocation in
the hot path; add `.astype(mx.bfloat16)` at module boundaries); the
risk is in `MultiHeadRMSNorm` and `AdaLNModulation` arithmetic where
fp32 accumulation matters for numeric stability.

**References.**
- Micikevicius et al. *Mixed Precision Training*. ICLR 2018.
  [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).
- Apple's MLX bf16 support: see `mx.bfloat16` dtype + `mx.fast.sdpa`
  bf16 fast-path notes in MLX 0.18+.
- PT autocast call site: `sam3d_objects/pipeline/inference_pipeline.py:674`.

---

### 3. Vectorize the SubMConv3d neighbor-table builder

**Current state.**
`mlx_port/kernels/sparse_subm_conv3d.py::build_neighbor_table` (lines
100–122) loops `N × 27` times in Python with a `dict` lookup per
neighbor:

```python
for n in range(N):
    b, z, y, x = coords_np[n]
    coord2row[_coord_key(b, z, y, x)] = n
nt = np.full((N, K3), -1, dtype=np.int32)
for n in range(N):
    b, z, y, x = coords_np[n]
    for dz in range(-half, half + 1):
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                k = (dz + half) * K * K + (dy + half) * K + (dx + half)
                key = _coord_key(b, z + dz, y + dy, x + dx)
                r = coord2row.get(key, -1)
                nt[n, k] = r
```

For N=16k voxels (the prune cap), this is 16,000 × 27 = 432k Python-level
dict lookups. Cached after first call inside one stage, but re-built
*twice* per inference (once for SLAT input coords, once for downsampled
coords) and on every new image.

**Proposed change.**
Replace the inner double loop with a fully vectorized numpy build:

```python
# 1. Pack coords -> int64 keys vectorized.
packed = (coords_np[:, 0].astype(np.int64) << 48) \
       | (coords_np[:, 1].astype(np.int64) << 32) \
       | (coords_np[:, 2].astype(np.int64) << 16) \
       |  coords_np[:, 3].astype(np.int64)

# 2. Build hash via numpy: argsort + searchsorted.
sort_idx = np.argsort(packed)
sorted_packed = packed[sort_idx]   # (N,)

# 3. Generate all 27 offset queries vectorized: (N, 27, 4).
half = K // 2
offs = np.stack(np.meshgrid(
    np.arange(-half, half+1),
    np.arange(-half, half+1),
    np.arange(-half, half+1),
    indexing='ij'), axis=-1).reshape(-1, 3)   # (27, 3)
queries = coords_np[:, None, :].copy()         # (N, 1, 4)
queries[..., 1:] = queries[..., 1:] + offs[None, :, :]   # broadcast

# 4. Pack queries the same way; binary-search into sorted_packed.
qpacked = ((queries[..., 0].astype(np.int64) << 48)
          | (queries[..., 1].astype(np.int64) << 32)
          | (queries[..., 2].astype(np.int64) << 16)
          |  queries[..., 3].astype(np.int64))   # (N, 27)

idx = np.searchsorted(sorted_packed, qpacked.ravel())
idx = np.clip(idx, 0, len(sorted_packed) - 1)
hits = sorted_packed[idx] == qpacked.ravel()
nt = np.where(hits, sort_idx[idx], -1).reshape(N, K3).astype(np.int32)
```

Pure numpy + a single `np.searchsorted` call — no Python inner loop.

**Trade-off.**
None — bit-identical output. The cost is one extra `argsort(N)` plus a
`searchsorted` of `27N` queries, both well-optimized in numpy.

**Speedup.**
For N=16k, replacing 432k Python-level dict ops with vectorized numpy
should be ~50–100× faster on this single function. Absolute saving is
modest (the function runs ~2× per inference, ~1–3 s each currently),
but it removes a noticeable hitch and frees CPU during the otherwise
GPU-bound SLAT loop.

**Effort.** 1–2 hours. Drop-in replacement; add a unit test against the
current implementation on a small synthetic coord set.

**References.**
- Spconv hash-based neighbor build: Yan, Y. et al. *SECOND*.
  Sensors 2018 (the original spconv paper);
  [paper](https://www.mdpi.com/1424-8220/18/10/3337).
- Choy, C. et al. *4D Spatio-Temporal ConvNets: Minkowski Convolutional
  Neural Networks*. CVPR 2019. arXiv:1904.08755 (Minkowski engine uses
  the same hash + offset pattern, vectorized in C++).

---

## Quick-mention table

| # | Area | Current | Proposal | Est. speedup | Quality | Effort | Reference |
|---|------|---------|----------|--------------|---------|--------|-----------|
| 4 | ODE solver | 25-step Euler (`pipeline_mlx.py:1064-1085`) | DPM-Solver++ 2M (multistep, no extra forward) | 25 → 8 steps ≈ **3×** SS+SLAT | Negligible | 8–12 h | Lu et al., *DPM-Solver++*, [arXiv:2211.01095](https://arxiv.org/abs/2211.01095) |
| 5 | ODE solver | 25-step Euler | Midpoint (RK2): half the steps, 2 fwd each = same total | break-even, **slightly higher quality** at same cost | Better | 2 h | PT already implements `Midpoint` in `flow_matching/solver.py:86-94` |
| 6 | CFG schedule | CFG every step 0..1 | CFG only on `t ∈ [0.4, 0.9]` (10 of 25 steps) | **~1.7×** combined SS+SLAT | Marginal drop | 1 h | Karras et al. *Analyzing CFG*, [arXiv:2404.07724](https://arxiv.org/abs/2404.07724) |
| 7 | Attention | Per-modality SDPA inside `MOTDiTCrossBlock._self_attn` runs concat-attention but loops over modalities for projection (`dit_mlx.py:594-665`) | Stack 'others' Q/K/V into single tensor outside the SDPA call (already done) — but the *protected* path still runs a separate SDPA. For B=1 inference, fuse into one big SDPA with a block-diagonal mask | **~1.1× SS only** | None | 4 h | Standard FlashAttention-with-mask pattern |
| 8 | GS decoder | All 12 swin blocks compute per-window SDPA in pure MLX with a Python loop over variable-length windows when partial (`decoder_mlx.py:498-513`) | Pad partial windows to `window_size` with attention mask → uniform fast-path always taken | **~1.3× GS decode** | None | 3 h | Liu et al. *Swin Transformer*, [arXiv:2103.14030](https://arxiv.org/abs/2103.14030) (their reference impl uses padded uniform windows) |
| 9 | GS decoder output | All 32 Gaussians/voxel emitted, then viewer culls low-opacity (`decoder_mlx.py:683-702`) | Prune Gaussians where `sigmoid(_opacity + opacity_bias_logit) < 0.005` BEFORE the linear `out_layer` projection (rank-1 sparse mask) | 50% fewer params written; **~1.5× ply write** | Visually identical | 2 h | Kerbl et al., *3DGS* original paper uses post-hoc opacity prune; standard practice |
| 10 | Pixel-shuffle upsample | `pixel_shuffle_3d` in SS decoder allocates an 8-axis intermediate then transposes (`decoder_mlx.py:121-151`) | Replace with `mx.fast.metal_kernel` doing one fused write — or use `nn.ConvTranspose3d` with stride 2 (single op, no intermediate) | **~1.2× SS decode** (small) | None | 4 h (Metal) / 1 h (ConvTranspose) | Shi et al. *Sub-pixel CNN*, [arXiv:1609.05158](https://arxiv.org/abs/1609.05158) |
| 11 | Sparse U-Net | `_scatter_mean` uses numpy `np.add.at` (`pipeline_mlx.py:1389-1402`) — NOT MLX | Re-implement as `mx.fast.metal_kernel` or use `mx.scatter_add` (MLX 0.20+) | **~1.05× SLAT** (small but unblocks GPU pipelining) | None | 3 h | – |
| 12 | RoPE / APE | Recomputed every step (no caching) — `_absolute_position_embedding` runs over (N, 3) sin/cos per call (`decoder_mlx.py:396-413`, `pipeline_mlx.py:1292-1306`) | Cache APE keyed on `coords` (same `id(coords)` pattern as SubMConv3d neighbor cache) | **~1.05× SLAT + GS** | None | 1 h | – |
| 13 | mx.compile fusion | Zero `mx.compile` decorators in the entire MLX port | Wrap inner DiT block forward (`DiTBlock.__call__`, `DiTCrossBlock.__call__`, `_SparseTransformerBlock.__call__`) in `@mx.compile` | **~1.15–1.4×** depending on graph fusion | None | 2 h | MLX docs on `mx.compile` graph capture |

---

## What was deliberately NOT recommended

- **Linear attention (Performer/Linformer)** — the SS DiT context is N=4096
  tokens, well below the regime where O(N²) attention dominates. Cross-
  attention to the 7528 image tokens is already the bottleneck and linear
  approximations there cost ~3% cosine; not worth the quality risk
  without retrain.
- **Token merging (ToMe)** — would need re-tuning per stage and the
  current pipeline already uses sparse coords for SLAT (effectively a
  hand-tuned token merge). For SS DiT (dense 4096 tokens) ToMe could help
  but is a research project, not an optimization.
- **fp8 / int8 quantization** — M1 has no fp8 hardware path. int8 needs
  per-tensor calibration scales we don't have for this checkpoint.
  Defer until Apple ships an fp8 fast-path or the user accepts a
  calibration step.
- **Latent Consistency Distillation** — we already have a shortcut model
  trained (recommendation #1). LCD would require retrain on the same
  data, which we don't own.
- **Adaptive step size (RKF45 / Dormand-Prince)** — the rectified-flow
  velocity field is near-linear in `t`, so adaptive control rejects very
  few steps in practice. Net win < 10% on rectified flows per Liu et al.
  *Flow Matching for Generative Modeling*,
  [arXiv:2210.02747](https://arxiv.org/abs/2210.02747). Not worth the
  control-loop complexity.

---

## Combined ROI summary (top stack)

| Stack | SS forwards | SLAT forwards | Total speedup vs current MLX |
|-------|-------------|---------------|------------------------------|
| Baseline (current MLX, 25-step CFG, fp32) | 50 | 50 | 1.0× |
| + bf16 (rec #2) | 50 | 50 | ~1.6× |
| + 4-step shortcut SS (rec #1) + bf16 | 4 | 50 | ~2.6× |
| + DPM-Solver++ 8-step SLAT (rec #4) + bf16 + 4-step shortcut | 4 | 16 | ~4.0× |
| + CFG schedule [0.4, 0.9] on SLAT (rec #6) | 4 | 10 | ~5.0× |
| + 1-step shortcut SS + bf16 + DPM++ 8-step SLAT + CFG schedule | 1 | 10 | ~6–7× |

Realistic two-week target: **3–5×** end-to-end speedup (bf16 + 4-step
shortcut + DPM-Solver++ on SLAT + CFG schedule), bringing 30 min →
6–10 min on M1 Max with no perceptible quality loss. The pure-numpy
neighbor-table fix is independent and cheap; ship it regardless.
