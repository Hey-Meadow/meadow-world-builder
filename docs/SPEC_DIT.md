# SPEC_DIT.md — Agent OBJ-DIT (Dense DiT backbone)

## Goal
Port the dense DiT (Diffusion Transformer) backbone from `sam3d_objects/model/backbone/tdfy_dit/` to MLX.

## Inputs
- **PT source**: `sam3d_objects/model/backbone/tdfy_dit/modules/transformer/`, `modules/attention/`, `model/layers/llama3/`
- **Plan**: `mlx_port/docs/PORT_PLAN.md`
- **Recon**: `mlx_port/docs/RECON_DIT.md` (read first — has the full module hierarchy)
- **Reference (already ported, READ to learn patterns)**:
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/mlx_port/models/vit_mlx.py` — ViT pre-norm transformer block (drop-in for DiT base)
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/mlx_port/tests/compare.py` — validation utility
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/mlx_port/weights/load_helpers.py` — loader pattern

## Out of scope
- Sparse blocks (octree, sparse attention) — handled by OBJ-METAL-SPARSE
- Training-only modules (loss, distillation)

## Key architectural elements (from RECON_DIT.md)
- **Pre-norm transformer**: 100% reusable from SAM 3D Body
- **AdaLN modulation**: time/conditioning embeddings produce scale+shift+gate per block — NEW vs ViT, port carefully
- **RoPE (rotary positional embedding)**: rotates Q,K — NEW vs ViT
- **RMS norm**: variant of LayerNorm — easy port
- **Multiple attention backends in PT**: xformers, flash_attn, SDPA — use only `mx.fast.scaled_dot_product_attention` in MLX

## Required deliverables

### 1. `mlx_port/models/dit_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn

class RoPE(nn.Module):
    """Rotary positional embedding."""
    def __init__(self, dim, max_pos=4096): ...
    def __call__(self, q: mx.array, k: mx.array, positions: mx.array) -> tuple[mx.array, mx.array]: ...

class AdaLNModulation(nn.Module):
    """time/cond embedding → (scale, shift, gate) × 2 (for attn and mlp)"""
    def __init__(self, cond_dim, hidden_dim): ...
    def __call__(self, cond: mx.array) -> dict: ...

class DiTBlock(nn.Module):
    """Pre-norm block with AdaLN modulation:
        x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * mlp(modulate(norm2(x), shift_mlp, scale_mlp))
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_rope=True): ...
    def __call__(self, x: mx.array, cond: mx.array, rope: RoPE = None,
                 positions: mx.array = None) -> mx.array: ...

class DiTBackbone(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio, cond_dim, use_rope=True): ...
    def __call__(self, x: mx.array, cond: mx.array,
                 positions: mx.array = None) -> mx.array: ...
    @classmethod
    def from_npz(cls, npz_path: str, prefix: str = "_base_models.generator.") -> "DiTBackbone":
        """Load from npz produced by mlx_port/weights/convert.py."""
        ...
```

### 2. `mlx_port/tests/test_dit.py`
- Random inputs (latent + cond), check that MLX DiT runs without error.
- Output a forward pass timing on M1 GPU (target: < 100ms per block batch=1).
- If you can find a way to capture PT reference outputs (e.g., one-time on cloud GPU or by noting that the body decoder has similar architecture), compare diff < 1e-3 fp32. Otherwise, just smoke test.

## Weights
- Two npz files to support: `mlx_port/weights/sam3d_objects/ss_flow.npz` (Stage 1 shape DiT, 3.3 GB) and `mlx_port/weights/sam3d_objects/slat_flow.npz` (Stage 2 texture DiT, 2.0 GB).
- Use `convert.py`'s output prefix structure (probably under `_base_models.generator.`).
- Inspect npz keys first:
  ```bash
  python -c "import mlx.core as mx; d = mx.load('mlx_port/weights/sam3d_objects/ss_flow.npz'); print(sorted(d)[:30])"
  ```

## Constraints
- MLX 0.31.2 (`mx.fast.metal_kernel`, `mx.fast.scaled_dot_product_attention` available)
- NHWC convention for any 4D inputs (we flatten anyway in DiT)
- Sequence layout `(B, N, C)` 
- fp32 default; document fp16 path for later
- Use `mx.fast.scaled_dot_product_attention` (don't write custom attention)
- DO NOT modify `sam3d_objects/` or `sam-3d-body/` (read-only)
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python` (already has MLX, torch, numpy)

## Definition of done
1. `dit_mlx.py` imports and `from_npz('ss_flow.npz')` loads without error.
2. Forward pass on random latent (B=1, N=4096, dim=1280) + cond (B=1, cond_dim=1280) runs.
3. Per-block forward < 100ms on M1 GPU.
4. Brief report (≤ 200 words) including:
   - npz key prefix structure observed
   - Per-block forward timing
   - Any non-trivial decisions (e.g., RoPE implementation choice)
   - Anything that needs to be coordinated with other agents
