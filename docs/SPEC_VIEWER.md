# SPEC_VIEWER.md — Agent OBJ-VIEWER

## Goal
Build a simple Gaussian splat .ply preview tool so we can quickly inspect output quality once INTEG produces splat.ply.

## Inputs
- 3DGS .ply standard format (per `mlx_port/models/decoder_mlx.py` `save_gaussian_ply`)
- Per-Gaussian fields: position(3), normals(3), f_dc(3), f_rest(45), opacity(1), scale(3), rotation(4)
- Reference renderer: `sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py` (PT, CUDA-only)

## Required deliverables
Pick ONE of these approaches (whichever you can ship cleanest):

### Option A (preferred): Web viewer
- `mlx_port/scripts/view_ply.py` — opens .ply in a browser via local http server using SuperSplat/three.js or a simple WebGL splat shader
- Just needs `python view_ply.py splat.ply` to spawn browser

### Option B: cv2 turntable renderer
- `mlx_port/scripts/render_ply.py` — CPU/MLX-based simple ortho-projection of Gaussian centers + colors
- Output: `preview.png` (front view) and `turntable.mp4` (spin around y-axis 360°, 60 frames)
- Doesn't have to be photorealistic — just "can I see object shape and color"

### Option C: minimal point cloud preview
- Render only Gaussian centers as points (ignore opacity, scale, rotation)
- Output PNG via Open3D (if installed) or matplotlib

## Strict rules
- DO NOT modify `sam3d_objects/`
- For Option A: use existing libraries, don't reinvent splatting
- For Option B/C: pure Python + numpy/cv2 OK (this is utility, not inference hot path)

## Definition of done
1. Given a valid 3DGS .ply, produces visible output in < 10 seconds
2. Test on a sample 3DGS .ply if you can find one online; otherwise generate a synthetic one (random Gaussians)
3. Report (≤200 words) which option you chose and why, plus how to invoke

## Constraints
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`
