# SPEC_LATENT_MAP.md — Agent OBJ-LATENT-MAP

## Goal
Port `latent_mapping` modules — the per-modality input/output projections (and pos_emb) that connect EmbedderFuser output → DiTBackbone input, and DiT output → Decoder input.

## Why this is needed
OBJ-DIT report: "OBJ-EMBED owns `latent_mapping.*` (per-modality input/output projection + pos_emb) — not loaded here." OBJ-EMBED finished the embedders but did NOT touch `latent_mapping`. This module sits between EMBED and DIT.

## Inputs
- **PT source (read these first)**:
  - `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py` (354 lines)
  - `sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py` (290 lines)
  - Look for class members named `latent_mapping`, `input_proj`, `output_proj`, `pos_emb`
- **npz weights** (already converted): `mlx_port/weights/sam3d_objects/{ss_flow.npz, slat_flow.npz}` — keys NOT prefixed with `reverse_fn.backbone.` likely belong here
- **Reference**:
  - `mlx_port/models/dit_mlx.py` (OBJ-DIT output) — DiT expects `(B, N, dim=1024)` input
  - `mlx_port/models/embedders_mlx.py` (OBJ-EMBED output) — produces `(B, N_cond, 1024)` conditioning tokens, but ALSO needs the latent path

## Required deliverables

### 1. `mlx_port/models/latent_mapping_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn

class LatentMapping(nn.Module):
    """Per-modality input projection: latent (B, N, in_dim) -> (B, N, dim).
    For MOT (multi-modality) ss_flow, this is a ModuleDict keyed by modality."""
    @classmethod
    def from_npz(cls, weights_dict, prefix: str) -> "LatentMapping": ...
    def __call__(self, x: mx.array, modality: str = None) -> mx.array: ...

class PositionalEmbedding(nn.Module):
    """Learned absolute positional embedding (APE) — pe_mode='ape' per DiT report."""
    def __call__(self, x: mx.array, positions: mx.array = None) -> mx.array: ...

class OutputMapping(nn.Module):
    """DiT output (B, N, dim) -> velocity prediction in latent space (B, N, out_dim)."""
    @classmethod
    def from_npz(cls, weights_dict, prefix: str) -> "OutputMapping": ...
    def __call__(self, x: mx.array, modality: str = None) -> mx.array: ...
```

### 2. `mlx_port/tests/test_latent_mapping.py`
- Load real npz keys
- Smoke test forward shape match
- For ss_flow MOT case: test all modalities (`shape`, `6drotation_normalized`, etc.)

## Strict MLX rules
- **Pure MLX, no numpy/torch in hot path**
- Use `mx.fast.layer_norm` if applicable, `mx.einsum` for projections
- nn.Linear is fine (it's MLX-native)
- DO NOT modify `sam3d_objects/` or `sam-3d-body/`

## Definition of done
1. Module loads from npz, all `latent_mapping.*` and `pos_emb` keys consumed
2. Forward pass shape correct (verified by inspection or PT comparison)
3. Report (≤200 words):
   - Exact npz prefix structure observed
   - Whether MOT modality routing matches DiT's expectation
   - Any unexpected modules (e.g., extra norm, dropout — note for inference disable)

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
