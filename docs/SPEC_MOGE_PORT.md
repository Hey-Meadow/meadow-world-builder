# SPEC_MOGE_PORT.md — MoGe MLX port (single agent)

## Goal
Port the MoGe pointmap model to MLX so we can replace the dummy synthetic pointmap with real depth predictions, restoring SAM 3D Objects output quality.

## Why this matters
INTEG ran end-to-end with dummy pointmap → splat.ply produced but noisy (no recognizable object). Real MoGe pointmap should yield ~10-30k voxels (vs 250k corrupted) and clean geometry.

## Inputs
- **PT source**:
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/lib/python3.11/site-packages/moge/model/v1.py` (390 lines) — `MoGeModel` class
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/lib/python3.11/site-packages/moge/model/dinov2/` — DINOv2 backbone (timm-style)
  - `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/lib/python3.11/site-packages/moge/utils/geometry_torch.py` (322 lines) — `recover_focal_shift`
- **Existing MLX DINOv2 port (REUSE)**:
  - `/Users/akaihuangm1/Desktop/github/lingbot-map/lingbot_map_mlx/dinov2_vit.py`
- **Existing weights**: HF `Ruicheng/moge-vitl` (309.9M params, 1.26 GB fp32)
- **Reference reconnaissance**: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/mlx_port/docs/MOGE_RECON.md` (already written, has line numbers + decision rationale)

## Architecture (304.4M backbone + 5.6M head)
- **Backbone**: DINOv2 ViT-L/14 (no register tokens variant in MoGe — verify) → reuse `lingbot_map_mlx.dinov2_vit`
- **Head**: small conv decoder producing pointmap + mask
- **Post-process**: `recover_focal_shift` solves for focal length and z-shift via least squares

## Required deliverables

### 1. `mlx_port/models/moge_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn

class MoGeModel(nn.Module):
    """DINOv2 + Head → pointmap (3, H, W) + mask (1, H, W)"""
    
    @classmethod
    def from_pretrained(cls, model_id: str = "Ruicheng/moge-vitl") -> "MoGeModel":
        """Download from HF, convert PT state_dict → MLX, cache locally."""
        ...
    
    def __call__(self, image: mx.array) -> dict:
        """
        image: (B, H, W, 3) preprocessed RGB in [0, 1] (or ImageNet-normalized)
        returns: {
            "points": (B, 3, H, W) — pointmap (xy in image plane, z = depth),
            "mask":   (B, 1, H, W) — valid mask,
            "intrinsics": (B, 3, 3) — recovered camera intrinsics,
        }
        """
        ...

def infer_pointmap(image_rgba: np.ndarray, mask: np.ndarray, model: MoGeModel = None) -> np.ndarray:
    """
    Helper: take RGBA + mask → run MoGe → return pointmap (3, H, W) in
    pytorch3d camera convention, ready to feed into SAM 3D Objects pipeline
    via pipeline.run(image, pointmap=this).
    Lazy-loads MoGeModel.from_pretrained() on first call (singleton).
    """
    ...
```

### 2. Weight conversion
- Either: write `mlx_port/weights/convert_moge.py` mirroring existing `convert.py` pattern
- Or: convert inside `from_pretrained` (download .pt, transpose Conv weights for NHWC, save as npz once, reuse)
- Cache npz at `mlx_port/weights/moge/moge_vitl.npz`

### 3. `mlx_port/tests/test_moge.py`
- Smoke test: synthetic image → forward pass → check pointmap shape (3, H, W), mask shape (1, H, W), intrinsics (3, 3)
- Real test: load kidsroom image + mask, run MoGe, verify pointmap z values are sensible (positive depth in mask region)

### 4. INTEG plumbing update
- Modify `mlx_port/models/pipeline_mlx.py` to swap `_synthetic_pointmap()` → `MoGeModel.from_pretrained()` for real pointmap
- Add CLI flag `--use-moge / --dummy-pointmap` to `mlx_port/infer_mlx.py`
- Run end-to-end with real MoGe → confirm voxel count drops from 250k to expected ~10-30k

## Strict MLX rules
- Pure MLX, no torch in inference hot path
- Use `mx.fast.scaled_dot_product_attention` (DINOv2 attention)
- For DINOv2: REUSE `lingbot_map_mlx.dinov2_vit` (don't reimplement)
- DO NOT modify `sam3d_objects/`, `sam-3d-body/`, or `lingbot-map/`

## Definition of done
1. `MoGeModel.from_pretrained()` downloads + converts + caches npz
2. Forward pass on kidsroom image produces sensible pointmap (verified by visualization or numerical sanity checks: positive z in mask region, NaN outside)
3. End-to-end pipeline run with real MoGe completes successfully (`mlx_port/infer_mlx.py --use-moge`)
4. New `splat.ply` produced, voxel count significantly < 250k
5. Re-render via `mlx_port/scripts/render_ply.py` and compare PNG to dummy-pointmap baseline
6. Report (≤300 words):
   - Per-stage timing: MoGe forward + total pipeline
   - New voxel count + Gaussian count
   - Visual quality assessment (open the new turntable mp4)
   - Any HF download / weight conversion issues

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python` (already has MoGe PT installed for reference)
- Allow ~1-2 hours for download + conversion + testing
