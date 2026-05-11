# SPEC_MOGE_RECON.md — Agent OBJ-MOGE-RECON

## Goal
Investigate the `moge` module that PT SAM 3D Objects uses to generate pointmaps. Determine if it's:
(a) optional — pipeline runs fine with `pointmap=None`
(b) required, small, MLX-portable
(c) required, large, NOT portable — recommend a fallback (constant z, random z, etc.)

## Inputs
- **PT pipeline**: `sam3d_objects/pipeline/inference_pipeline_pointmap.py` (579 lines)
- `notebook/inference.py` — passes `pointmap=None` by default; signature has `pointmap` param
- `WHITELIST_FILTERS` mentions `moge` module
- Pipeline yaml: `checkpoints/hf/pipeline.yaml`
- moge package on disk: search `/Users/akaihuangm1/Desktop/github/sam-3d-objects/.venv/lib/python*/site-packages/moge/` or pip show moge

## Required deliverables
1. **`mlx_port/docs/MOGE_RECON.md`** report covering:
   - Where moge is invoked in PT pipeline (line numbers)
   - Whether `pointmap=None` short-circuits cleanly or if pipeline always computes one
   - moge model architecture (size in params, ViT? DINO-based?)
   - moge model weights (size on disk)
   - **Decision**: which of (a)/(b)/(c) above
   - If (b): port estimate (LoC, complexity)
   - If (c): proposed fallback strategy (e.g., constant depth z=0, image-aligned xy)

## Constraints
- Pure investigation — DO NOT write any inference code
- DO NOT modify `sam3d_objects/`
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
