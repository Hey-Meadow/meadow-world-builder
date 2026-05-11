# SPEC_EVAL.md — Agent OBJ-EVAL

## Goal
Build evaluation script skeleton: takes a list of (image, mask) pairs, runs the MLX pipeline (when ready), and produces summary stats (timing, Gaussian count, file size, success/fail).

## Inputs
- Future pipeline interface (per `mlx_port/docs/SPEC_INTEG.md`):
  ```python
  from mlx_port.models.pipeline_mlx import SAM3DObjectsPipeline
  p = SAM3DObjectsPipeline.from_npz_dir(...)
  output = p(image, seed=42)
  output["gs_params"]  # dict of MLX arrays
  output["voxels"]     # (N, 3) coords
  ```
  (If pipeline_mlx.py is not yet ready, use a stub interface and document)
- Test images: `notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png` + masks 0..N
  (find what masks exist via: `ls notebook/images/shutterstock_stylish_kidsroom_1640806567/*.png`)

## Required deliverables

### 1. `mlx_port/scripts/eval_pipeline.py`
```python
"""
Usage:
  python mlx_port/scripts/eval_pipeline.py --image-dir DIR [--out-dir OUT] [--seeds 42,43] [--max-cases 10]

Produces:
  OUT/case_{name}_seed_{s}/splat.ply
  OUT/eval_summary.csv  (case, seed, total_sec, ss_sec, slat_sec, n_gaussians, ply_bytes, status)
  OUT/eval_summary.json (full structured)
"""
```

### 2. Image discovery helper
- Scan for `image.png` + `*.png` masks pattern under directory tree
- Produce list of (image_path, mask_path) tuples for each mask index found

### 3. Reporting
- After eval: print summary table (mean/p50/p95 timing, success rate)
- Compare seeds: do same image+mask pairs with different seeds give different output sizes?

## Strict rules
- DO NOT modify `sam3d_objects/` or `mlx_port/models/` (read-only consumer of pipeline)
- This script is a USER tool not inference hot path — numpy/pandas OK
- Graceful degradation: if pipeline raises, log error, write status=fail to summary, continue with next case

## Definition of done
1. Script runs even without pipeline (use mock/stub pipeline that returns dummy splat for testing the script logic)
2. CSV + JSON summary written
3. Test on at least 3 mock cases to verify aggregation logic
4. Report (≤150 words): summary table format, what columns, how to extend later

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
