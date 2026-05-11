# SPEC_EMBED.md — Agent OBJ-EMBED (Conditioning embedders)

## Goal
Port conditioning embedders to MLX:
- Image embedder (DINOv2/v3 ViT)
- Time embedding (sinusoidal + MLP)
- Point patch embedder (for pointmap conditioning)
- Embedder fuser (combine multiple modalities)

## Inputs
- **PT source**: `sam3d_objects/model/backbone/dit/embedder/`, `sam3d_objects/model/backbone/dit/`
- **Plan**: `mlx_port/docs/PORT_PLAN.md`
- **Reference (already ported, can REUSE)**:
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/mlx_port/models/vit_mlx.py` — full ViT (use as DINO base)
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/mlx_port/models/dinov3_mlx.py` — DINOv3 wrapper
- **Weights**: 
  - `mlx_port/weights/sam3d_objects/ss_embedder.npz` (2.2 GB)
  - `mlx_port/weights/sam3d_objects/slat_embedder.npz` (2.2 GB)

## Required deliverables

### 1. `mlx_port/models/embedders_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding + 2-layer MLP."""
    def __init__(self, dim: int, max_period: int = 10000): ...
    def __call__(self, t: mx.array) -> mx.array: ...

class ImageEmbedder(nn.Module):
    """DINO ViT trunk → patch tokens. Reuses sam-3d-body ViT pattern."""
    def __init__(self, **vit_cfg): ...
    def __call__(self, image: mx.array) -> mx.array: ...

class PointPatchEmbedder(nn.Module):
    """Encode pointmap (3 channels of 3D coords) similar to image patches."""
    def __init__(self, patch_size: int, embed_dim: int, in_channels: int = 3): ...
    def __call__(self, pointmap: mx.array) -> mx.array: ...

class EmbedderFuser(nn.Module):
    """Combine image + pointmap + (optional) class label into conditioning tokens."""
    def __init__(self, ...): ...
    def __call__(self, *modalities) -> mx.array: ...

class ConditionEmbedder(nn.Module):
    """Top-level container that wires Image + Point + Time + Fuser."""
    @classmethod
    def from_npz(cls, npz_path: str, prefix: str = "_base_models.condition_embedder.") -> "ConditionEmbedder": ...
```

### 2. `mlx_port/tests/test_embedders.py`
- Random RGBA image (B=1, H=518, W=518, C=4) — produces conditioning tokens
- Random time (B=1) → time embedding
- Smoke test: produces tokens of expected shape

## Inspection step
Before coding, run:
```bash
python -c "
import mlx.core as mx
d = mx.load('mlx_port/weights/sam3d_objects/ss_embedder.npz')
keys = sorted(d.keys())
print('total:', len(keys))
print('first 30:'); [print(k) for k in keys[:30]]
print('last 10:'); [print(k) for k in keys[-10:]]
"
```
This reveals the actual key prefixes (DINO, fuser, etc.) so you can match them.

## Constraints
- Pure MLX
- Reuse `sam-3d-body/mlx_port/models/vit_mlx.py` as DINO trunk (don't reimplement)
- DO NOT modify `sam3d_objects/` or `sam-3d-body/` (read-only)
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`

## Definition of done
1. `embedders_mlx.py` imports + `ConditionEmbedder.from_npz('ss_embedder.npz')` loads
2. Test runs without error
3. Output token shape matches expected (B, N_tokens, embed_dim)
4. Report (≤ 200 words): key structure + DINO variant identified + any unusual modules
