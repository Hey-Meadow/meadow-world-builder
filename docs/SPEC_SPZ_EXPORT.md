# SPEC_SPZ_EXPORT.md — Agent OBJ-SPZ-EXPORT

## Goal
Convert SAM 3D Objects' `.ply` (3DGS standard) output → Niantic `.spz` format (compressed Gaussian splat, ~10× smaller).

## Why this is needed
- `.ply` is uncompressed and large (34.82 MB for 512k Gaussians)
- `.spz` is Niantic's compact format for web/mobile delivery
- User wants both formats from the MLX pipeline

## Inputs
- **Reference**: https://github.com/nianticlabs/spz — official SPZ format spec + Rust/C++ tool
- **Input format**: 3DGS `.ply` produced by `mlx_port/models/decoder_mlx.py` `save_gaussian_ply`
  - 17 props per vertex: `x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..44 (or fewer for SH degree < 3), opacity, scale_0..2, rot_0..3`
  - Uses canonical activations: log-scale, sigmoid opacity, quaternion rotation
- **Test file**: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/splat.ply` (34.82 MB, 512k Gaussians)

## Approach options

### Option A: Use Niantic's spz Rust CLI (preferred if installable)
```bash
brew install spz   # or build from source
spz encode splat.ply splat.spz
```
- Wrap in Python helper for our pipeline
- Cleanest, uses official reference implementation

### Option B: Pure Python SPZ encoder
- Implement the SPZ format spec ourselves (it's documented in the repo)
- Required if Niantic CLI doesn't install on darwin-arm64

### Option C: Use existing Python bindings if any exist
- Search PyPI for `pyspz` / `spz-python` bindings

## Required deliverables

### 1. `mlx_port/scripts/ply_to_spz.py`
```python
"""
Convert 3DGS .ply (SAM 3D Objects output) → Niantic .spz format.

Usage:
  python mlx_port/scripts/ply_to_spz.py splat.ply [--out splat.spz]
"""
```

### 2. Test on the existing `splat.ply`
- Verify output size is significantly smaller (~3-5 MB expected for 512k Gaussians)
- Verify file is valid SPZ (decode-roundtrip if possible)

### 3. Pipeline integration
- Add `--format ply|spz|both` flag to `mlx_port/infer_mlx.py`
- When `spz` or `both`, also produce `splat.spz` alongside `.ply`

## Strict rules
- DO NOT modify `sam3d_objects/`
- This is utility tool, not inference hot path — numpy/struct/zlib all OK
- DO NOT bundle the entire Niantic spz repo — install via package manager or git clone outside the project tree

## Definition of done
1. `splat.ply` → `splat.spz` conversion works
2. Output `.spz` file size measured
3. Decode + render roundtrip if possible (use Niantic's spz decode tool to verify, or our own decoder)
4. CLI integration: `python mlx_port/infer_mlx.py --image X --mask Y --format both` produces both files
5. Report (≤200 words): which option (A/B/C) used, file size comparison, any quality loss

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
