# DiT Backbone Reconnaissance Report

## Overview
SAM 3D Objects uses two DiT (Diffusion Transformer) variants totaling ~12k LoC:
- **dit/**: Embedder-only (image+time+class embeddings), ~600 LoC
- **tdfy_dit/**: Full 3D-aware DiT with flow/VAE, ~10.3k LoC

Focus: TDFY-DiT is the main inference backbone (~340 depth on 512x384 images in 8-step flow matching).

## Module Hierarchy

```
tdfy_dit/ (10.3k LoC)
├── modules/ (2.6k)
│   ├── transformer/
│   │   ├── blocks.py (197)
│   │   │   ├── AbsolutePositionEmbedder [sinusoidal, 3D spatial]
│   │   │   ├── FeedForwardNet [Linear→GELU→Linear, mlp_ratio=4.0]
│   │   │   ├── TransformerBlock [pre-norm: norm1→attn→+, norm2→mlp→+]
│   │   │   └── TransformerCrossBlock [3 norms, self-attn, cross-attn, mlp]
│   │   └── modulated.py (341)
│   │       ├── ModulatedTransformerBlock [uses AdaLN, 6-param modulation]
│   │       ├── ModulatedTransformerCrossBlock [+ cross-attn variant]
│   │       └── MOTModulatedTransformerCrossBlock [multi-modality per-latent]
│   ├── attention/ (550)
│   │   ├── full_attn.py (176)
│   │   │   └── scaled_dot_product_attention() [multi-backend: xformers, flash_attn, sdpa, naive]
│   │   └── modules.py (374)
│   │       ├── MultiHeadRMSNorm [learned per-head scale]
│   │       ├── RotaryPositionEmbedder [RoPE, complex representation]
│   │       └── MultiHeadAttention [self/cross, rope optional, qk_rms_norm optional]
│   │           └── MOTMultiHeadSelfAttention [modality-protected routing]
│   ├── sparse/ (1.4k)
│   │   ├── attention/ (999)
│   │   │   ├── full_attn.py, masked_sdpa.py, serialized_attn.py, windowed_attn.py
│   │   │   └── modules.py [sparse tensor variants, fixed block layout]
│   │   ├── transformer/ (351)
│   │   │   ├── blocks.py [SparseTransformerBlock, SparseTransformerCrossBlock]
│   │   │   └── modulated.py [ModulatedSparseTransformerBlock]
│   │   ├── conv/conv_spconv.py [spconv ops, sparse convolution]
│   │   ├── basic.py, linear.py, nonlinearity.py, norm.py, spatial.py
│   │   └── [Uses minkowski.SparseTensor | spconv SparseConvTensor]
│   ├── norm.py (26) [LayerNorm32, GroupNorm32, ChannelLayerNorm32]
│   ├── spatial.py, utils.py
│   └── [Extras: render_utils, postprocessing_utils]
├── models/ (1.5k, inference-critical)
│   ├── sparse_structure_flow.py (303)
│   │   └── SparseStructureFlowModel [main flow-matching sampler]
│   ├── mot_sparse_structure_flow.py [multi-modality variant]
│   ├── structured_latent_flow.py [latent-space flow]
│   ├── sparse_structure_vae.py, mm_latent.py [training-only]
│   ├── structured_latent_vae/ (encoder, decoder_gs/mesh/rf) [VAE backbone]
│   └── timestep_embedder.py (sinusoidal + MLP)
└── [representations/, renderers/ — see Agent OBJ-2]

dit/ (600 LoC, embedder-only)
├── embedder/
│   ├── dino.py [frozen DINO feature backbone]
│   ├── embedder_fuser.py [fuse image+time+class]
│   ├── pointmap.py, point_remapper.py
│   └── __init__.py [entry points]
```

## Transformer Block Structure

### Standard Block (TransformerBlock)
- **Pattern**: Pre-norm residual
  ```
  x' = x + attn(norm1(x))
  x'' = x' + mlp(norm2(x'))
  ```
- **Norm**: LayerNorm32 (casts to float32 internally), eps=1e-6
- **Attention**: MultiHeadAttention, channels → channels
- **MLP**: Linear(C, C*4) → GELU → Linear(C*4, C)
- **Options**:
  - `use_rope=True`: rotary positional embeddings on Q, K
  - `qk_rms_norm=True`: MultiHeadRMSNorm post-projection
  - `attn_mode="full"` or `"windowed"` (windowed not yet implemented)
  - `use_checkpoint=True`: gradient checkpointing

### Modulated Block (AdaLN style)
- **Pattern**: Post-norm modulation (DiT paper style)
  ```
  h = norm1(x) * (1 + scale_msa) + shift_msa
  h = attn(h) * gate_msa
  x = x + h
  h = norm2(x) * (1 + scale_mlp) + shift_mlp
  h = mlp(h) * gate_mlp
  x = x + h
  ```
- **6-param modulation per block**: 2 × (shift, scale, gate)
- **Modulation source**: `mod` ← time/class embedding → SiLU → Linear(C, 6C)
- **share_mod=True**: reuses modulation across blocks (efficiency)

### Cross-Attention Block
- **3 submodules**: self-attn, cross-attn (context), mlp
- **Cross path**: K, V from context; Q from latent (full attention only)
- **qk_rms_norm_cross**: optional for cross branch

## Attention Variants

### MultiHeadAttention
- **Backends** (full_attn.py):
  - `xformers`: memory-efficient (meta only, proprietary)
  - `flash_attn`: fused kernel (cutlass-based)
  - `torch_flash_attn`: PyTorch native (v2.0+, bfloat16 only)
  - `sdpa`: torch.nn.functional (PyTorch 1.13+)
  - `naive`: manual dot-product (teaching only)
- **RoPE**: RotaryPositionEmbedder (complex sqrt-1 * phases)
- **QK-RMS-Norm**: MultiHeadRMSNorm per head dim, learned gamma scale

### Sparse Attention (3D-specific)
- **SparseTensor wrapper**: minkowski.SparseTensor or spconv.SparseConvTensor
- **Attention modes**:
  - `full_attn.py`: dense fallback if sparse coords fit
  - `masked_sdpa.py`: PyTorch SDPA with attention mask (structured sparsity)
  - `serialized_attn.py`: flattened token-by-token (memory-efficient)
  - `windowed_attn.py`: local 3D window attention
- **Sparse conv**: `spconv.Conv3d` wrappers for latent features

## Conditioning Mechanisms

### 1. Time Embedding
- **TimestepEmbedder**: sinusoidal → MLP(256 → hidden_size → hidden_size)
- **Injection**: → modulation vector (6 × channels per block)

### 2. Image Features
- **Source**: DINO backbone (dit/embedder/dino.py, frozen)
- **Integration**: cross-attention context in ModulatedTransformerCrossBlock

### 3. Class Labels / Multi-Modality
- **MOTMultiHeadSelfAttention**: per-modality Q/K/V projection (ModuleDict)
- **Protect_modality_list**: (default "shape") blocks gradient flow from non-protected modalities
- **Use case**: Gaussian splatting, mesh, radiance field outputs in parallel

## Sparse 3D Distinctions

**Which modules require sparse ops:**
1. **Sparse attention** (modules/sparse/attention/):
   - Handles octree / voxel grids (coordinate lists)
   - Serialization for >1M points (per-token loops)
2. **Sparse conv** (modules/sparse/conv/conv_spconv.py):
   - 3D convolution on octree (pre-flow or post-VAE)
3. **SparseTransformerBlock** (modules/sparse/transformer/blocks.py):
   - Wraps SparseTensor, serialized attention fallback
4. **Sparse structure flow** (models/sparse_structure_flow.py):
   - Main inference flow: octree → latent → flow steps → 3D output

**Not sparse:**
- Standard blocks (modules/transformer/) operate on dense tokens
- Latent space is always dense (B, T, C)

## Reusable Patterns from SAM 3D Body MLX Port

### Already ported (vit_mlx.py):
1. **Pre-norm blocks**: `Block` class mirrors TransformerBlock structure exactly
2. **Attention**: `Attention` class (self-attn only) → extend to cross-attn
3. **MLP**: `Mlp` class (Linear→GELU→Linear, mlp_ratio=4.0)
4. **Positional embeddings**: pos_embed tensor (no special MLX handling)
5. **Weight loading**: from_pytorch_state() pattern reusable

### Map to DiT (requires new):
| SAM 3D Body (ported) | SAM 3D Objects (new) | Notes |
|---|---|---|
| Block (pre-norm) | TransformerBlock | ✓ Same |
| Attention (self) | MultiHeadAttention (self) | Adds ROPE, RMS-norm, windowed |
| — | MultiHeadAttention (cross) | New, for image conditioning |
| — | ModulatedTransformerBlock | New, DiT-style AdaLN |
| — | SparseTensor attention | Requires spconv/minkowski replacement |
| Mlp | FeedForwardNet | ✓ Same |
| — | TimestepEmbedder | Simple sinusoidal + 2 Linear |

## MLX Port Effort Estimate (days, 1 engineer)

| Component | LoC | Effort | Notes |
|---|---|---|---|
| **Norm + utils** | 100 | 0.5 | LayerNorm32 → MLX native; trivial |
| **TransformerBlock + FeedForwardNet** | 200 | 0.5 | Copy Body patterns, test GELU |
| **MultiHeadAttention (self)** | 150 | 1.0 | RoPE, QK-RMS-norm new; sdpa → mx.fast |
| **MultiHeadAttention (cross)** | 50 | 0.5 | Extend self; key/value from context |
| **TimestepEmbedder** | 40 | 0.5 | Sinusoidal, trivial |
| **ModulatedTransformerBlock** | 300 | 1.5 | AdaLN scaling, gate multiplication |
| **RotaryPositionEmbedder** | 80 | 1.5 | Complex arithmetic, phases; test vs torch |
| **MultiHeadRMSNorm** | 20 | 0.5 | Parameter scaling, normalize |
| **SparseTransformerBlock** | 150 | 2–3 | **Blocker**: spconv → MLX is uncharted |
| **Sparse attention** | 500+ | 3–5 | **Blocker**: serialized/masked ops |
| **Weight loading** | 200 | 1.0 | Iterate on from_pytorch_state |
| **Integration test** | — | 1.0 | End-to-end single forward pass |
| **TOTAL** | ~1.8k | **10–13 days** | Dense blocks: ~3 days; sparse: 5–7 days |

## Known Blockers & Risks

1. **spconv / minkowski requirement**
   - Sparse 3D conv (spconv.SparseTensor / minkowski.SparseTensor)
   - No native MLX equivalent; would need full replacement (octree → dense latent ?)
   - **Risk**: Inference on octree 3D shapes may not port without rewriting sparse ops.

2. **Flash attention backends**
   - PyTorch supports xformers, flash_attn, sdpa, torch_flash_attn
   - MLX offers only mx.fast.scaled_dot_product_attention (SDPA-like)
   - **Risk**: Performance cliff vs cuda; no windowed attention in MLX
   - **Mitigation**: Use mx.fast for all, skip windowed variant (not implemented anyway)

3. **Complex number arithmetic (RoPE)**
   - PyTorch: torch.view_as_complex, torch.polar for easy phase rotation
   - MLX: limited; may need manual matmul + trig
   - **Risk**: Numerical divergence; would need careful porting + testing

4. **Multi-modality routing (MOTMultiHeadSelfAttention)**
   - Per-modality ModuleDict + torch._pytree.tree_map
   - MLX: no tree_map; must unfold manually or rewrite for dict handling
   - **Risk**: Code verbosity; moderate complexity

5. **Training-only code**
   - VAE encoders, loss functions, data loading in models/
   - **Mitigation**: Skip; focus on inference path (sparse_structure_flow.py)

## Summary for Planning

**Porting feasibility: MODERATE-TO-HIGH** (blocks & dense attention) **→ HIGH-RISK** (sparse 3D)

- **Dense transformer blocks**: 3–4 days, straightforward (reuse Body patterns)
- **Attention (self + cross + RoPE)**: 2–3 days, some numerical care needed
- **Sparse 3D and spconv**: **5–7 days or impractical** (no MLX sparse ops library)

**Recommendation**: Start with dense-only DiT (modules/transformer/, modules/attention/), defer sparse to Phase 2. This unblocks single-representation inference (Gaussian splatting or dense mesh latent). Sparse octree variant requires either:
- (A) Replace spconv with MLX tensor operations (moderate rewrite)
- (B) Keep octree processing in PyTorch, pipe dense latents to MLX flow (hybrid)
- (C) Simplify to dense voxel grid (loses octree compression advantage)

