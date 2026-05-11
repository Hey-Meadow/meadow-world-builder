# SPEC_LATENT_SHARE.md — Agent OBJ-LATENT-SHARE

## Goal
Port `latent_share_transformer` — the wrapper-level module that merges the 3 extra ss_flow modalities (`scale`, `translation`, `translation_scale`) into the 2 routed-through-DiT modalities (`shape`, `6drotation_normalized`).

## Why this is needed
OBJ-LATENT-MAP discovered: ss_flow has 5 modalities, but DiT only routes 2. The other 3 are merged via `latent_share_transformer` at the wrapper level. This merge is currently NOT ported. Without it, the SS flow path is incomplete.

## Inputs
- **PT source**:
  - `sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py` (290 lines)
  - Look for `latent_share_transformer`, `SparseStructureFlowTdfyWrapper`
- **npz weights**: `mlx_port/weights/sam3d_objects/ss_flow.npz` — keys with `latent_share_transformer.*` prefix
- **Already ported (read for context)**:
  - `mlx_port/models/dit_mlx.py` — MOTDiTBackbone (downstream consumer)
  - `mlx_port/models/latent_mapping_mlx.py` — LatentMapping (upstream producer)

## Required deliverables

### 1. `mlx_port/models/latent_share_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn

class LatentShareTransformer(nn.Module):
    """Merges 5 modality latents into 2 (shape, 6drotation_normalized) via cross-attention or concat+proj.
    
    Inputs:
        latents: dict of {modality: mx.array (B, N_i, 1024)} for all 5 modalities
    Outputs:
        merged: dict of {'shape': (B, N_shape, 1024), '6drotation_normalized': (B, N_rot, 1024)}
    """
    @classmethod
    def from_npz(cls, weights_dict, prefix: str = "reverse_fn.backbone.latent_share_transformer.") -> "LatentShareTransformer": ...
    def __call__(self, latents: dict) -> dict: ...

class LatentSplitTransformer(nn.Module):
    """Inverse direction at output: splits DiT output back into 5 modalities for OutputMapping."""
    @classmethod
    def from_npz(cls, weights_dict, prefix: str = "...") -> "LatentSplitTransformer": ...
    def __call__(self, merged: dict) -> dict: ...
```

### 2. `mlx_port/tests/test_latent_share.py`
- Load real npz keys
- Forward with synthetic 5-modality input → check 2-modality output shape
- Round-trip: split(merge(latents)) ≈ latents (if architecture allows)

## Strict MLX rules
- Pure MLX, no numpy/torch in inference hot path
- Use `mx.fast.scaled_dot_product_attention` for any cross-attention
- `mlx.nn.LayerNorm`, `nn.Linear` are fine
- DO NOT modify `sam3d_objects/` or `sam-3d-body/`

## Open questions for the agent to resolve
1. Is `latent_share_transformer` cross-attention or simple concat+linear merge? Read PT source.
2. Is there a separate split module at output, or does the DiT output go through reverse `LatentMapping` directly with only 2 modalities?
3. How are scale/translation handled — do they predict their own velocity, or do they just inform the shape DiT path?

## Definition of done
1. Module loads from npz, all `latent_share_transformer.*` keys consumed
2. Forward pass produces correct shape
3. Report (≤200 words): merge mechanism, output structure, open issues for INTEG

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
