# SPEC_INTEG.md — Agent OBJ-INTEG

## Goal
Wire all Phase 1+2 modules into an end-to-end MLX inference pipeline that reads a single RGBA image + mask and outputs a Gaussian splat .ply file.

## Inputs
- **PT reference (read this first to understand orchestration)**:
  - `sam3d_objects/pipeline/inference_pipeline_pointmap.py` (579 lines) — main pipeline
  - `sam3d_objects/pipeline/inference_pipeline.py` (846 lines) — base pipeline
  - `sam3d_objects/pipeline/inference_utils.py` (1104 lines) — helpers
  - `notebook/inference.py` — high-level Inference wrapper, shows demo entry
  - `demo.py` — entry point: load_image + load_single_mask → inference(image, mask, seed=42) → splat.ply
- **MLX modules already ported (in `mlx_port/models/`)**:
  - `embedders_mlx.py` — `ConditionEmbedder` (loads ss_embedder.npz, slat_embedder.npz)
  - `dit_mlx.py` — `DiTBackbone` (slat) + `MOTDiTBackbone` (ss) (loads ss_flow.npz, slat_flow.npz)
  - `sampler_mlx.py` — `FlowMatching` + `CFGWrapper`
  - `decoder_mlx.py` — `SSDecoder` + `SLATDecoderGS` + `save_gaussian_ply`
  - `latent_mapping_mlx.py` — `LatentMapping` + `OutputMapping` (OBJ-LATENT-MAP, may be in flight)
  - `sparse_blocks_mlx.py` — `SparseInputBlocks` + `SparseOutputBlocks` (OBJ-SPARSE-WRAPPER, may be in flight)
- **Test image**: `notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png` + mask `14.png`

## Pipeline architecture (from PT inference_pipeline_pointmap.py)

```
Input: RGBA image (H, W, 4)  — alpha channel = object mask

Stage 0: Image preprocess
  ├─ Resize to 518x518 for DINOv2
  ├─ ImageNet normalize
  └─ Generate pointmap (PT uses 'moge' module — investigate, may need to skip or use simple depth proxy)

Stage 1: SS (sparse structure) flow
  ├─ ConditionEmbedder.ss(image, pointmap) -> cond_tokens (1, 3764, 1024)
  ├─ Initialize random latent (B=1, N=initial_voxels, dim=8)
  ├─ FlowMatching.sample(MOT_DiT_backbone, cond, num_steps=25, cfg_strength=7.0)
  │     where backbone_fn wraps: latent_mapping_in -> MOTDiTBackbone -> latent_mapping_out
  ├─ SSDecoder(latent) -> 3D voxel coords (N_voxels, 3)
  └─ Output: sparse voxel grid

Stage 2: SLAT (sparse latent) flow
  ├─ ConditionEmbedder.slat(image) -> cond_tokens (1, 2740, 1024)
  ├─ Initialize random sparse latent at voxels from Stage 1
  ├─ FlowMatching.sample(DiT_backbone with sparse_input_blocks/out_blocks, cond, cfg_strength=5.0)
  ├─ SLATDecoderGS(latent, voxels) -> Gaussian params (N_voxels × 32 gaussians)
  └─ save_gaussian_ply(gs, "splat.ply")
```

## Required deliverables

### 1. `mlx_port/models/pipeline_mlx.py`
```python
import mlx.core as mx
from mlx_port.models import embedders_mlx, dit_mlx, sampler_mlx, decoder_mlx
from mlx_port.models import latent_mapping_mlx, sparse_blocks_mlx

class SAM3DObjectsPipeline:
    """End-to-end MLX inference pipeline."""
    
    @classmethod
    def from_npz_dir(cls, npz_dir: str = "mlx_port/weights/sam3d_objects/") -> "SAM3DObjectsPipeline":
        """Load all modules from npz files."""
        ...
    
    def __call__(
        self,
        image: mx.array,           # (H, W, 4) RGBA uint8
        seed: int = 42,
        ss_steps: int = 25,
        slat_steps: int = 25,
        ss_cfg: float = 7.0,
        slat_cfg: float = 5.0,
    ) -> dict:
        """Returns dict with 'voxels', 'gs_params', 'ply_bytes'."""
        ...
```

### 2. `mlx_port/infer_mlx.py` (CLI entry)
```python
"""
Usage:
  python mlx_port/infer_mlx.py --image PATH --mask PATH [--seed 42] [--out splat.ply]

Or RGBA-merged input:
  python mlx_port/infer_mlx.py --rgba PATH [--seed 42] [--out splat.ply]
"""
```

### 3. `mlx_port/tests/test_pipeline.py`
- Smoke test: run on `notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png` + mask 14
- Verify .ply file produced, valid format, non-zero Gaussian count
- Note: Even garbage output is fine for first pass — we just need it to RUN end-to-end

## Pointmap source (CRITICAL investigation)
PT uses `moge` module (visible in `WHITELIST_FILTERS` of inference.py). Action:
1. Read inference_pipeline_pointmap.py to see how pointmap is generated
2. Options:
   a. Pointmap is computed inside pipeline using a separate model (moge) → may need to MLX-port that too
   b. Pointmap is OPTIONAL — can pass `pointmap=None` and pipeline falls back to image-only conditioning
   c. Pointmap is required and not portable → use a simple depth heuristic for first pass (constant z, image-aligned xy)
**Priority**: prefer (b) if possible, fall back to (c) for first runnable end-to-end.

## Strict MLX rules
- **All inference hot path: pure MLX**
- Image I/O: PIL or `mx.io` (whichever works)
- Numpy ONLY for image loading/saving conversion at boundaries
- DO NOT modify `sam3d_objects/` or `sam-3d-body/` (read-only)

## Definition of done
1. `python mlx_port/infer_mlx.py --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png --mask notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png` runs to completion
2. Produces a valid `splat.ply` file (Gaussian count > 0, openable in MeshLab or any GS viewer)
3. End-to-end timing reported (target: < 5 minutes on M1 Max)
4. Report (≤300 words):
   - Pointmap source resolution (b/c above)
   - Per-stage timing breakdown
   - Any modules where `from_npz` failed or had key mismatches
   - Final Gaussian count and visual quality assessment (open the .ply if you can)

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
- Coordinate with OBJ-LATENT-MAP and OBJ-SPARSE-WRAPPER (in flight) — if their modules aren't ready, use stub interfaces and document
