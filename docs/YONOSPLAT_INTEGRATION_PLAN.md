# YoNoSplat MLX — integration plan (post-sprint)

The 9-agent parallel sprint shipped 8 ported sub-modules under `meadow_sb/`. This doc captures what's left to wire them into a working end-to-end encoder and the 8 interface-contract corrections discovered along the way.

## Sprint results — what landed

| Agent | Module | LOC (code + test) | Tests | Worst diff vs PT | Commit |
|---|---|---:|---:|---:|---|
| A | `dinov2_encoder.py` | 340 + 140 | 24/24 blocks (script) | 2.23e-3 (block 22) | `1c8c6a0` |
| B | `croco_decoder.py` | 474 + 153 | 37 (pytest) | 2.37e-5 | `8e029e1` (cherry-picked from worktree) |
| C | `sub_decoders.py` | 324 + 319 | 3 (pytest) | 3.24e-5 | `29cd397` |
| D | `heads.py` | 329 + ? | 5 (pytest, after conftest fix) | 1.24e-5 | `b8d47bd` |
| E | `gaussian_adapter.py` | 291 + ? | 6 fields, 1.19e-7 | 1.19e-7 | `f2c2346` |
| F | `rasterizer.py` (Tier-1 gsplat-CPU) | 432 + ? | 3 (pytest) | 0 (bit-exact) | `3f27872` |
| G | `convert_weights.py` | ? | 5 (pytest) | — | bundled in `f2c2346` (concurrency capture) |
| H | `e2e_test.py` + `RUNPOD_REFERENCE_PLAN.md` | 611 + 236 | 1 (pytest) | — | `d86fdfd` |
| I | `utils/` (mlx_helpers, weight_loader, attention) | 844 | 4 (pytest) | 0 (bit-exact) | `19bc87f` |

**Total**: ~5 600 LOC code + ~1 200 LOC tests + 62 passing tests. End-to-end `e2e_test.py` reports 8/8 module families discovered.

## 8 interface-contract corrections discovered by the sprint

The bootstrap-phase `YONOSPLAT_INTERFACE_CONTRACT.md` was written from state-dict inspection only. Live code reading by the parallel agents found these mistakes:

| # | Was | Actually | Impact |
|---|---|---|---|
| 1 | DINOv2 has 5 register tokens | **4 register tokens** in the ViT; the 5th lives in the Pi3 decoder | Token count is 261 = 1 cls + 4 reg + 256 patch |
| 2 | CroCo decoder has cross-attention, 54 tensors/block | **Self-attn only + RoPE2D + qk_norm**, 18 tensors/block. Cross-view fusion via **reshape** (even blocks per-view, odd blocks concat-view) | Big simplification — saves writing cross-attn code |
| 3 | `PointHead` is `Linear(1024, 3)` | `Linear(1024, 147)`; the 3 emerges from `pixel_shuffle` (147 = 3·7²) | Caller must apply pixel_shuffle, not the head |
| 4 | `RgbEmbed` is `Conv2d(3, 1024, 14, 14)` | `Conv2d(3, **2048**, **7**, 7)` → output (B·V, 1024, 2048) | Patch size is 7×7 not 14×14 |
| 5 | `CameraHead` input is 1024 | **Input is 512** — `camera_decoder` projects 1024→512 before the head | Camera decoder out_dim ≠ point/gaussian out_dim |
| 6 | Sub-decoders depth 8 | **Depth 5**, 12 tensors/block. State-dict: 5 × 12 + 4 = 64 ✓ | Smaller than expected |
| 7 | `GaussianAdapter` scale = `sigmoid · (max − min) + min` | Upstream uses `UnifiedGaussianAdapter` = `0.001 · softplus(x).clamp_max(0.3)` — config min/max unused | Match the Unified path |
| 8 | Opacity from a separate branch | **Channel 0 of the 539-d** gaussian head output; caller `sigmoid` + unsqueeze before passing the 10-d remainder into the adapter | xyz still comes from `point_head` (correct) |

All eight have been baked into the agent code; `INTERFACE_CONTRACT.md` should be amended to match.

## Concurrency observation

Worktree isolation only worked for Agents A, B, C (their commits landed on their `worktree-agent-*` branches). Agents D, E, F, G, H, I all wrote into the master checkout, so their commits landed directly on `feature/space-port-research`. Side effect: Agent E's commit `f2c2346` accidentally captured Agent G's staged files (E and G ran simultaneously). Code is correct; commit attribution is a bit wrong. No code conflicts because each agent wrote to distinct files.

For future parallel sprints: use `Agent({isolation: "worktree"})` *and* explicitly tell agents to stay in their worktree (some agents may have `cd` ed out).

## What's left (integration sprint — next ~3-5 days)

### 1. Top-level assembler (`meadow_sb/models/yonosplat.py`)

Skeleton already committed but `__call__` raises `NotImplementedError`. The blocker is **factory consistency**: each agent's module exposes a slightly different loader entry-point. Standardise:

- `from .dinov2_encoder import load_encoder_from_state_dict` — present
- `from .croco_decoder import build_croco_decoder_from_state_dict` — TBD verify
- Same pattern for sub-decoders, heads, etc.

Once all sub-modules expose `*_from_state_dict()` the assembler's `__call__` is a 60-LOC chain of:
```
imgs → backbone → upsample → rgb_embed_add → 3 parallel sub-decoders →
  3 heads → SO(3) project camera → GaussianAdapter → return
```

### 2. End-to-end numerical validation

Compare `YoNoSplatEncoder(weights)(images, intrinsics)` against the upstream PT path on the same `test_input.npz`:

- Gaussians.xyz max abs diff < 1e-3
- Gaussians.scale, rotation, opacity, features each < 1e-3
- camera_poses max abs diff < 1e-3
- intrinsic_pred max abs diff < 1e-4

This is the **real** quality gate that subsumes all per-module gates.

### 3. SVD orthogonalisation

Camera head outputs a 9-d unconstrained matrix → must project to SO(3). MLX 0.31 has no `mx.linalg.svd`, so we round-trip through numpy in `svd_orthogonalise()`. The 3×3 matrix size makes this a non-issue performance-wise (B·V × 9 floats × one SVD).

### 4. Rasterizer glue

Agent F's `GsplatRasterizer.render()` takes torch tensors (gsplat-CPU is PyTorch). The boundary at `infer.py`:

```
adapter_out (mx.array)  →  np.array(...)  →  torch.from_numpy  →  rasterizer  →  rendered RGB
```

Pure-CPU path, no MLX/CUDA collision. Validated by Agent H.

### 5. RunPod reference run

`docs/RUNPOD_REFERENCE_PLAN.md` (Agent H) has the exact A100 commands. Cost ≈ $1–3. Outputs to download:

- `rendered.png` — CUDA-rendered ground truth
- `gaussians.ply` — final Gaussian splat for SuperSplat inspection
- `timing.json` — per-stage A100 wall time

Validate our MLX end-to-end output against these.

### 6. Speed benchmark on M1 Max

After assembler lands, run the full pipeline on the `test_input.npz` to capture per-stage timing on Apple Silicon. Document in a `BENCHMARK_YONOSPLAT.md` matching the existing `meadow_wb/docs/FINAL_BENCHMARK.md` template.

## Decision point — repo split

Sprint outputs landed under `meadow_sb/` in this repo (`meadow-world-builder`). When v0.0.1 of YoNoSplat-MLX is ready to ship, decide:

1. Spin into separate `Hey-Meadow/meadow-space-builder` GitHub repo (clean, but loses the shared `meadow_wb/utils/`)
2. Keep as a sibling package within `meadow-world-builder` (current state, more code reuse)

Recommend **option 2 until v0.1.x** — defer the split until tooling like `convert_weights.py`, `web/` viewer, GIF gallery, etc. need duplication.

## Risks still on the table

| Risk | Likelihood | Mitigation |
|---|---|---|
| DINOv2 block-22 fp32 drift (2.23e-3 vs 1e-3 gate) propagates through stack | Medium | Bump assembler to bf16 mixed precision in the encoder; verify end-to-end still passes |
| Some agent's `load_*_from_state_dict()` doesn't exist or is named differently | High (probability we'll need to standardise) | Day-1 of integration sprint: enumerate and align signatures |
| gsplat-CPU at 92 ms/render × 224² is the speed floor — too slow for production | Medium | Tier-2 Metal rasterizer is the planned fix (separate ~2-3 week sprint) |
| SO(3) round-trip through numpy adds Apple-Silicon-specific quirks | Low | Tiny operation, cheap to validate |

## Next concrete action

Run a single forward pass through the assembler-with-numpy-fallbacks on `test_input.npz` and compare against `dumps/per_block/backbone_full.npz`. Whichever sub-module's `load_*` call breaks first becomes the day-1 task for the integration follow-up.
