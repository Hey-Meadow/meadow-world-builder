# SPEC_DECODER.md — Agent OBJ-DECODER (Sparse latent → Gaussian splat decoder)

## Goal
Port the decoders that turn sparse latent tokens into Gaussian splat parameters (xyz, scale, quat, opacity, SH coeffs).

## Inputs
- **PT source**: 
  - `sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/` (Gaussian model)
  - `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/` (decoders)
- **Plan**: `mlx_port/docs/PORT_PLAN.md`
- **Recon**: `mlx_port/docs/RECON_3D.md` (read first — Gaussian decoder is "no SparseConv3d, just SparseLinear + scatter ops")
- **Weights**:
  - `mlx_port/weights/sam3d_objects/ss_decoder.npz` (166 MB) — sparse structure decoder (latent → 3D voxel coords)
  - `mlx_port/weights/sam3d_objects/slat_decoder_gs.npz` (193 MB) — sparse latent → Gaussian params
  - `mlx_port/weights/sam3d_objects/slat_decoder_gs_4.npz` (191 MB) — high-res variant

## Key insight from recon (RECON_3D.md)
> slat_decoder_gs uses NO SparseConv3d — only SparseLinear (= nn.Linear on flat features) plus sparse Downsample/Upsample/Subdivide which are pure torch (scatter_reduce, gather, cumsum). Decoder is Mac-portable as-is.

This means: **no Metal kernel needed for decoder**. Pure MLX with `mx.gather` + `mx.scatter_add` should work.

## Required deliverables

### 1. `mlx_port/models/decoder_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn

class SparseLinear(nn.Module):
    """Just nn.Linear on (N, in_dim) -> (N, out_dim) sparse features."""
    ...

class SparseSubdivide(nn.Module):
    """For each voxel, produce 8 children at next octree level. Pure index manipulation."""
    ...

class SparseUpsample(nn.Module):
    """Scatter-based upsample of features."""
    ...

class SSDecoder(nn.Module):
    """Latent (B, N, dim) -> 3D voxel coords (predicted occupancy)."""
    @classmethod
    def from_npz(cls, npz_path: str) -> "SSDecoder": ...

class SLATDecoderGS(nn.Module):
    """Sparse latent + voxel coords -> per-voxel Gaussian splat params.
    
    Output: dict with
      - xyz:       (N, 3)
      - scale:     (N, 3)
      - rotation:  (N, 4)  quaternion
      - opacity:   (N, 1)
      - sh_coeffs: (N, 48) for degree 3 SH (up to 16 coeffs × 3 channels)
    """
    @classmethod
    def from_npz(cls, npz_path: str) -> "SLATDecoderGS": ...

# Helper: pack Gaussian params into .ply file (3DGS format)
def save_gaussian_ply(gs_params: dict, path: str) -> None: ...
```

### 2. `mlx_port/tests/test_decoder.py`
- Smoke test with random sparse latents → check Gaussian output shapes
- Optional: load real npz weights, run forward pass (won't be meaningful without real input from upstream, but proves it loads)

## Sparse data structures
Per recon, ops needed:
- `mx.gather(features, indices)` — gather sparse features by index
- `mx.scatter_add(dst, indices, src)` — scatter add for upsample
- `mx.cumsum(...)` — running prefix sum for octree subdivide

If MLX lacks any, fall back to indexing tricks. Document in report.

## Output: Gaussian splat .ply file
- 3DGS standard format (open in any GS viewer)
- Per-Gaussian: position(3), normals(3, ignored=0), f_dc(3, base color SH), f_rest(45, SH 1-3), opacity(1), scale(3), rotation(4)
- See https://github.com/graphdeco-inria/gaussian-splatting for spec

## Constraints
- Pure MLX, no torch in inference path
- DO NOT modify `sam3d_objects/` or `sam-3d-body/` (read-only)
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`

## Definition of done
1. Both decoder classes import + `from_npz` loads
2. Forward pass on random input produces correct-shaped output (gaussian params dict)
3. `save_gaussian_ply` produces a valid .ply file (open in viewer if you can; even MeshLab works)
4. Report (≤ 200 words) including:
   - Number of Gaussians produced (default config)
   - Output structure
   - Any sparse op MLX lacks + workaround
