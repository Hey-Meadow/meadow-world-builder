# SPEC_SPARSE_WRAPPER.md — Agent OBJ-SPARSE-WRAPPER

## Goal
Wrap the slat_flow's `input_blocks` and `out_blocks` (sparse 3D conv blocks at the input/output of SLAT DiT) using `SparseSubmConv3d` from OBJ-METAL-SPARSE.

## Why this is needed
OBJ-DIT report: "OBJ-METAL-SPARSE owns slat_flow's `input_blocks` / `out_blocks` (sparse 3D conv) — not loaded here."
OBJ-METAL-SPARSE delivered the SubMConv3d kernel only — no wrapper for input_blocks/out_blocks structure.

## Inputs
- **PT source**:
  - `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py` (354 lines) — find `input_blocks`, `out_blocks`, `down_blocks`, `up_blocks`
  - `sam3d_objects/model/backbone/tdfy_dit/modules/sparse/` — sparse module library used by these blocks
- **npz weights**: `mlx_port/weights/sam3d_objects/slat_flow.npz` — keys with `input_blocks.*` or `out_blocks.*` prefix
- **Reference**:
  - `mlx_port/kernels/sparse_subm_conv3d.py` (OBJ-METAL-SPARSE output) — `SparseSubmConv3d` class
  - `mlx_port/models/dit_mlx.py` — slat_flow DiT expects flat `(B, N, 1024)` after input_blocks

## Required deliverables

### 1. `mlx_port/models/sparse_blocks_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn
from mlx_port.kernels.sparse_subm_conv3d import SparseSubmConv3d

class SparseConvBlock(nn.Module):
    """Standard sparse 3D conv block: SubMConv3d + GroupNorm + activation.
    Used by input_blocks/out_blocks. Pure MLX."""
    ...

class SparseInputBlocks(nn.Module):
    """Stack of sparse conv blocks at DiT input.
    sparse_features (N, in_ch) + coords (N, 3) -> latent_tokens (N, dim) for DiT."""
    @classmethod
    def from_npz(cls, weights_dict, prefix: str = "reverse_fn.input_blocks.") -> "SparseInputBlocks": ...

class SparseOutputBlocks(nn.Module):
    """Stack of sparse conv blocks at DiT output.
    DiT output (N, dim) -> sparse latent (N, out_ch)."""
    @classmethod
    def from_npz(cls, weights_dict, prefix: str = "reverse_fn.out_blocks.") -> "SparseOutputBlocks": ...
```

### 2. `mlx_port/tests/test_sparse_blocks.py`
- Random sparse input (N=1000, coords in 0..32 grid)
- Forward through input_blocks → check shape becomes DiT-compatible
- Forward through out_blocks → check shape matches expected latent

## Strict MLX rules
- **Pure MLX, no numpy/torch in hot path**
- Reuse `SparseSubmConv3d` for all 3D conv ops
- For GroupNorm/SiLU: use `mlx.nn` built-ins
- Cache neighbor table at sample-load time (submanifold optimization — same coords across all 50 ODE steps)

## Definition of done
1. `SparseInputBlocks.from_npz(slat_flow_weights, "reverse_fn.input_blocks.")` loads, all keys consumed
2. `SparseOutputBlocks.from_npz(slat_flow_weights, "reverse_fn.out_blocks.")` loads, all keys consumed
3. Forward pass produces correct shapes
4. Report (≤200 words):
   - Exact block stack structure (depth, channel progression)
   - Number of npz keys consumed under input_blocks/out_blocks prefix
   - Whether neighbor-table caching is wired correctly (build once, reuse 50× per ODE)

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
