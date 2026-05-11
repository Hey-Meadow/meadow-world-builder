# YoNoSplat MLX Port — Interface Contract

Generated during the bootstrap phase on Apple M1 Max from the released `re10k_224x224_ctx2to32.ckpt` checkpoint. All shapes and naming verified by direct introspection of the upstream PyTorch checkpoint, **not paper-derived**. Parallel agents porting individual modules use this as the single source of truth — input/output shape, dtype, and weight-key naming.

## Where to find local artifacts

| Path | Content |
|---|---|
| `research/yonosplat_bootstrap/weights/pi3/model.safetensors` | Pi3 pre-trained backbone (3.83 GB) |
| `research/yonosplat_bootstrap/weights/yonosplat/re10k_224x224_ctx2to32.ckpt` | YoNoSplat re10k finetune (3.86 GB, 965 M params) |
| `/tmp/yonosplat_inspect/` | Upstream code (shallow clone, MIT) for cross-reference |

> The shallow clone in `/tmp/` is volatile; pin the upstream commit before agent work begins.

## Top-level state-dict layout (1222 tensors)

| Prefix | Tensor count | Role |
|---|---:|---|
| `encoder.backbone` | 1002 | Pi3 backbone: encoder + decoder + intrinsic head |
| `encoder.point_decoder` | 64 | 3-D point regression sub-net (8 blocks × 8 tensors) |
| `encoder.gaussian_decoder` | 64 | Gaussian feature regression sub-net (8 blocks × 8 tensors) |
| `encoder.camera_decoder` | 64 | Camera token decoder (8 blocks × 8 tensors) |
| `encoder.camera_head` | 20 | Camera pose head (MLP + projection) |
| `encoder.rgb_embed` | 4 | RGB pixel embedding (small Conv) |
| `encoder.point_head` | 2 | Final point output projection |
| `encoder.gaussian_head` | 2 | Final Gaussian output projection (1024 → 539) |

**Total: 965 M params · 3.86 GB fp32 · ~1.93 GB fp16**

## Backbone (Pi3) — verified shapes

### Encoder (DINOv2-ViT-L style)

- 24 transformer blocks
- Hidden dim **1024**
- Patch size **14 × 14**, RGB → 1024-dim tokens via `Conv2d(3, 1024, 14, 14)`
- Each block has **14 tensors**

```
encoder.backbone.encoder.patch_embed.proj.weight        : (1024, 3, 14, 14)
encoder.backbone.encoder.blocks.{0..23}.attn.qkv.weight : (3072, 1024)
encoder.backbone.encoder.blocks.{0..23}.attn.proj.weight: (1024, 1024)
encoder.backbone.encoder.blocks.{0..23}.mlp.fc1.weight  : (4096, 1024)
encoder.backbone.encoder.blocks.{0..23}.mlp.fc2.weight  : (1024, 4096)
encoder.backbone.encoder.blocks.{0..23}.norm1.weight    : (1024,)
```

### Decoder (CroCo cross-view, 12 blocks)

Cross-attention between views — this is where multi-view fusion happens.

```
encoder.backbone.decoder.{0..11}.attn.qkv.weight  : (3072, 1024)
encoder.backbone.decoder.{0..11}.attn.proj.weight : (1024, 1024)
...
```

### Intrinsic head

Small MLP predicting camera intrinsics from backbone tokens.

## Heads (the YoNoSplat-specific bit)

| Sub-net | Architecture | Input → Output |
|---|---|---|
| `encoder.point_decoder` | 8-block transformer | 1024 tokens → 1024 tokens (point-aware) |
| `encoder.gaussian_decoder` | 8-block transformer | 1024 tokens → 1024 tokens (GS-aware) |
| `encoder.camera_decoder` | 8-block transformer | 1024 tokens → 1024 tokens (camera-aware) |
| `encoder.point_head` | Linear | 1024 → 3 (xyz per token) |
| `encoder.gaussian_head` | Linear | 1024 → **539** (per-token Gaussian parameter vector) |
| `encoder.camera_head` | MLP | 1024 → 6 / 7 (pose params per view) |
| `encoder.rgb_embed` | Tiny Conv | RGB (3) → 1024 token augmentation |

### What is 539?

539 = `num_surfaces × (3 xyz + 3 scale + 4 quaternion + 1 opacity + 3·SH_count colour)`. Likely `num_surfaces = 7` with low-order SH. Read `gaussian_adapter.py` for exact split.

## Forward-pass contract (text-level)

```
Inputs
  images        : float32, (B, V, 3, 224, 224)
  intrinsics    : float32, (B, V, 3, 3) | None
  near, far     : float32, (B, V) | scalar

Encoder.backbone (Pi3)
  patch_embed       : (B·V, 3, 224, 224) → (B·V, 256, 1024)     # N = 16·16 tokens
  encoder (24 ViT) :  N tokens → N tokens
  decoder (12 CroCo): cross-attention between views, (B·V, N, 1024)

Heads (3 parallel branches)
  point_decoder    → point_head   : (B·V, N, 1024) → (B·V, N, 3)
  gaussian_decoder → gaussian_head: (B·V, N, 1024) → (B·V, N, 539)
  camera_decoder   → camera_head  : (B·V, N, 1024) → (B, V, pose_dim)

Outputs
  gaussians.xyz       : (B, V·N·S, 3)
  gaussians.scale     : (B, V·N·S, 3)
  gaussians.rotation  : (B, V·N·S, 4)
  gaussians.opacity   : (B, V·N·S, 1)
  gaussians.features  : (B, V·N·S, C)              # SH colour
  camera (R, t) per view
  intrinsics (B, V, 3, 3)
```

## Parallel-agent dispatch plan

| Agent | Module scope | LOC est. | Reference artifacts needed |
|---|---|---:|---|
| **A**: DINOv2 ViT encoder | 24-block self-attn, patch_embed, PE | ~1500 | per-block in/out tensors + patch_embed_out |
| **B**: CroCo cross-view decoder | 12-block self-attn + cross-attn | ~700 | decoder_in, decoder_out, per-block qkv |
| **C**: Sub-decoder transformers | 8 blocks × 3 paths (point/gauss/cam) | ~800 | sub-decoder in/out × 3 |
| **D**: Output heads | gaussian/point/camera/intrinsic + rgb_embed | ~300 | head in/out for each |
| **E**: GaussianAdapter | 539-vec → `Gaussians` struct | ~200 | adapter in/out tensors |
| **F**: Metal rasterizer (Tier 2) | forward-only splat | ~500 Metal + glue | gaussians_in, rendered_rgb |
| **G**: Weight conversion script | PT keys → MLX npz | — | full state_dict.pt |
| **H**: E2E test harness | wire + compare to RunPod ref | — | end-of-pipeline .ply |

Quality gate per agent: `max(|mlx_out − pt_ref|) < 1e-4` for fp32, `< 1e-2` for bf16.

## Open questions (resolve before parallel sprint)

1. `num_surfaces` exact value — read `gaussian_adapter.py` (likely 7).
2. SH degree — affects colour count per Gaussian.
3. `camera_head` pose dim — 6 / 7 / 9?
4. Patch positional embedding — RoPE or learnt?
5. Gradient checkpointing — toggle at inference?

All resolvable in ~30 min of upstream-code reading; assign as bootstrap-agent's first step.

## Local-vs-RunPod split

| Task | Where | Why |
|---|---|---|
| Activation-dump per block | Local CPU/MPS | rasterizer stub, cheap iteration |
| `.ply` opens correctly in SuperSplat | Local | encoder runs without CUDA |
| Rendered image PSNR vs GT | RunPod A100 | needs `diff_gaussian_rasterization` |
| Clean speed baseline | RunPod A100 | CUDA-native |
| Apple-Silicon timing | Local M1 Max | actual target |

## Component-wise param breakdown (from state-dict)

| Component | Tensors | Params | Bytes |
|---|---:|---:|---:|
| `encoder.backbone.decoder` (12 cross-view blocks) | 648 | **453.6 M** | **1.81 GB** |
| `encoder.backbone.encoder` (24 ViT blocks) | 343 | 304.4 M | 1.22 GB |
| `encoder.point_decoder` (8 blocks) | 64 | 66.1 M | 264 MB |
| `encoder.gaussian_decoder` (8 blocks) | 64 | 66.1 M | 264 MB |
| `encoder.camera_decoder` (8 blocks) | 64 | 65.6 M | 262 MB |
| `encoder.backbone.intrinsics_embed_layer` | 4 | 5.0 M | 20 MB |
| `encoder.camera_head` | 20 | 2.1 M | 8 MB |
| `encoder.backbone.intrinsic_head` | 4 | 1.0 M | 4 MB |
| `encoder.gaussian_head` | 2 | 0.55 M | 2 MB |
| `encoder.rgb_embed` | 4 | 0.31 M | 1 MB |
| `encoder.point_head` | 2 | 0.15 M | 0.6 MB |
| `encoder.backbone.register_token` | 1 | 0.01 M | — |
| `encoder.backbone.image_mean/std` | 2 | 0 | — |
| **Total** | **1222** | **965 M** | **3.86 GB** |

**Load-bearing surprise**: the **CroCo decoder is 1.5× more expensive than the DINOv2 encoder** (453 M vs 304 M params). Each decoder block has 54 tensors vs the encoder's 14 — cross-attention (extra Q, K, V, proj plus auxiliary norms) inflates per-block param count by ~3×. The Metal port will need careful attention here; this is where multi-view fusion happens.

## Resolved open questions (from `config/model/encoder/yonosplat.yaml` + source)

1. **num_surfaces = 1** — one Gaussian per token, not 7. Token-to-Gaussian unrolling happens via `upscale_token_ratio=2` + `gaussians_per_axis=14`.
2. **SH degree = 0** → `d_sh = 1` (DC only, no view-dependent colour).
3. **Camera pose dim = 12** — 9-dim rotation matrix (`fc_rot`) + 3-dim translation (`fc_t`). Not quaternion.
4. **Positional embedding = 2D sin/cos (MAE-style)**, not RoPE. RoPE inside CroCo is an internal detail.
5. **`use_checkpoint: true`** is training-only (gradient checkpointing memory savings); inference path skips.

The **539-dim gaussian_head output** is unpacked into a 2-D grid via the upscale head (each token → ~49 Gaussians × 11-dim before adapter remaps). Exact arithmetic to be confirmed when porting the GaussianAdapter — bake the formula into the agent prompt.

## Status

- ✅ Upstream code cloned, MIT licence confirmed
- ✅ Pure-Python deps installed (`lightning`, `gsplat`, `e3nn`, `lpips`, `dacite`, …)
- ✅ `diff_gaussian_rasterization` stub written → encoder + decoder import cleanly
- ✅ Weights downloaded (Pi3 + re10k.ckpt, ~7.7 GB total)
- ✅ State-dict structure verified (1222 tensors, 965 M params)
- ✅ Component-wise breakdown captured (see table above)
- ✅ **5 open questions resolved** (see above)
- ✅ Test input saved: `dumps/test_input.npz` (2-view 224×224 RGB)
- ✅ Tensor-key map saved: `dumps/state_dict_tensor_map.json` (148 KB; agents can grep here instead of loading the 3.8 GB checkpoint)
- ⏳ Real forward + per-block activation hook (`dump_pi3.py`) — needed before parallel-agent kickoff
- ⏳ Parallel-agent dispatch (8 agents per dispatch plan above)
