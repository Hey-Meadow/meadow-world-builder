# SAM 3D Objects → MLX Port Plan (REVISED)

## Strategy
**Skip PyTorch-on-Mac baseline.** Go directly to MLX + custom Metal kernels.
We replace each CUDA-only op with a Metal kernel rather than CPU stub.

## Why this is better than going through PyTorch baseline

- PT baseline requires patching `spconv`, `gsplat`, `pytorch3d`, `kaolin` with stubs — work we throw away.
- MLX port is the goal anyway; no detour.
- Validation comes from comparing tensor outputs against reference data captured ONCE on a CUDA machine (cloud GPU spot instance or Meta web demo), not from running full PT pipeline locally.
- We already have proven MLX patterns from SAM 3D Body port.

## Hard scope cuts
1. **Output: Gaussian splats only** (.ply) — skip mesh / texture / rendering
2. **Disable CFG by default** — 2× faster
3. **Dense DiT first**, sparse blocks via custom Metal in Phase 2
4. **No PyTorch dependency at runtime** — pure MLX + Metal

## Phase plan

### Phase 0: Preparation (in progress)
- ✅ OBJ-RECON: 3 recon agents → architecture mapped
- ✅ OBJ-WEIGHTS: checkpoint discovered, converter scaffolded
- ⏳ **BLOCKED: HF access for `facebook/sam-3d-objects`** (user must request)
- ❌ ~~PT-Mac baseline~~ (cancelled — go straight to MLX)

### Phase 1: Module ports + Metal kernels (parallel agents)
After HF access + weight conversion, dispatch 6 agents:

- **OBJ-DIT**: Dense DiT backbone in MLX (modulated AdaLN, RoPE, attention, MLP)
  - Reuse: `sam-3d-body/mlx_port/models/vit_mlx.py` patterns
- **OBJ-SAMPLER**: Flow matching Euler ODE + CFG wrapper in MLX
- **OBJ-EMBED**: Conditioning embedders (image → DINO, time, point patch, depth)
- **OBJ-DECODER**: Sparse latent → Gaussian splat parameters decoder
- **OBJ-METAL-SPARSE**: Metal kernel for sparse conv3d (replace spconv)
- **OBJ-METAL-GSPLAT**: Metal kernel for Gaussian splat rasterization (replace gsplat)
- **OBJ-G**: Test harness (reuse SAM 3D Body's `compare.py`)

### Phase 2: Integration
- **OBJ-INTEG**: Wire all phase 1 modules → end-to-end MLX inference
- **OBJ-REFERENCE**: Capture PT reference data ONCE (one-time cloud GPU run) for validation

## Output: Gaussian splat .ply file
- Per object: ~10k Gaussians × {xyz, scale_3d, quat, opacity, SH coeffs}
- Saved as standard 3DGS .ply format
- Can be viewed in any 3DGS viewer or converted to .spz with Niantic's tool

## Inference target
- M1 Pro: 30-90 sec/object (without CFG: 15-45 sec)
- Memory: < 12 GB peak (fits 16 GB Mac)

## Reuse from SAM 3D Body
- Test harness `tests/compare.py` (copy)
- Weight converter `weights/convert.py` (extended)
- ViT/transformer block patterns
- `mx.fast.scaled_dot_product_attention` usage
- Custom Metal kernel pattern via `mx.fast.metal_kernel`

## Validation strategy (without PT-on-Mac)

### Per-module validation
1. Save PT reference outputs for key modules (one-time, on cloud GPU):
   - DiT block forward: random latent + time embed → output
   - Embedder: random image → tokens
   - Decoder: random latent → Gaussian params
2. Run MLX equivalents on same inputs, compare max_abs_diff.
3. Tolerance: 1e-3 fp32, 1e-2 fp16.

### End-to-end validation  
1. Run MLX pipeline on the kidsroom test image.
2. Save .ply output.
3. Open in viewer → does it look like a recognizable 3D object?
4. (Optional) Compare to Meta's web demo output qualitatively.

## File structure
```
mlx_port/
├── docs/             ← plans + recon
├── models/           ← MLX module ports
│   ├── dit_mlx.py
│   ├── sampler_mlx.py
│   ├── embedders_mlx.py
│   └── decoder_mlx.py
├── weights/          ← npz + converter
├── kernels/          ← Metal kernels (replacing CUDA libs)
│   ├── sparse_conv3d.metal + sparse_conv3d_kernel.py
│   └── gaussian_splat.metal + gaussian_splat_kernel.py
├── tests/            ← per-module + end-to-end
├── reference_data/   ← PT outputs from one-time cloud capture
└── infer_mlx.py      ← CLI entry
```

## Out of scope
- PyTorch local baseline (skipped)
- pytorch3d, kaolin, nvdiffrast (skipped)
- Flexicubes mesh extraction (skipped)
- ShortCut distillation (training only)
- Texture baking (skipped)
