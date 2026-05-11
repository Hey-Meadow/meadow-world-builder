# Multi-view 3D Gaussian Splat merger — prototype report

Branch: `feature/multi-view-merge`
Script: `meadow_wb/scripts/merge_views.py`
Status: **prototype** — passes a synthetic ground-truth test, real multi-view fusion is an open problem (see Limitations).

## Goal

Take N `.ply` files (each produced by `infer.py` on a different view of the same object), align them into a single coordinate frame, concatenate the Gaussians, and write a merged `.ply`.

## Approach

Open3D **tensor-API ICP** with a **multi-start yaw-rotation grid** as a poor-man's global initialiser. The legacy `o3d.pipelines.registration.registration_ransac_*` API segfaults on Apple-Silicon Open3D 0.18 (verified: SIGSEGV before any work, even on canonical FPFH/RANSAC examples), so we replace it with a deterministic alternative:

1. **Downsample** each input PLY to `voxel_size` (default 0.02, ~0.5 cm for normalised object PLYs)
2. **Estimate normals** for point-to-plane ICP
3. For each `i ≥ 1`, register view `i` to view `0` by trying N initial yaw rotations (default 8, every 45°), running coarse-then-fine ICP from each, and keeping the highest-fitness result
4. **Transform** each view by its recovered T (also rotates Gaussian rotation quaternions via `quat * R_T`)
5. **Concatenate** all Gaussian rows, rewrite the PLY header `element vertex` count

## Synthetic test — PASS

Reference: `assets/demos/chair_clean.ply` (63 540 Gaussians, normalised to ~0.7-unit extent).

| Step | Detail |
|---|---|
| view 0 | chair_clean.ply, unchanged |
| view 1 | chair_clean.ply rotated 30° around Y + translated (0.2, 0.05, −0.1) |
| run | `merge_views.py view0.ply view1.ply --voxel-size 0.015 --rotation-hypotheses 8` |

Result:

```
view 1 → view 0:  fitness=0.524  inlier_rmse=0.0042  (3.6 s, 8 starts)
merged.ply: 127 080 Gaussians (8.6 MB)
```

Verification — nearest-neighbour distance from view 1's transformed points back to view 0's original points:

```
mean   = 0.0006
median = 0.0006
p95    = 0.0007
```

Error / object size ≈ 0.1 %. **Essentially perfect alignment.** ICP fitness 0.524 is misleading: it's measured against the *downsampled* version and reflects coverage, not accuracy.

## Real multi-view test — DEFERRED

Single-image 3DGS outputs are in *object-centric* frames whose orientation depends on the input camera angle. Two views of the same object captured from very different angles produce PLYs in fundamentally different frames — point-only ICP cannot recover the mapping without either:

1. **Initial pose estimate** from image-side (DUSt3R / MASt3R / COLMAP)
2. **Image-conditioned 3DGS fusion** trained jointly on multi-view inputs (the original 3DGS approach)
3. **Strong shape prior** that survives view changes (FPFH on geometric descriptors — but the legacy RANSAC that does this segfaults on ARM)

We don't currently have multi-angle photo sets of the same object to test option-1, and option-2 is a different project entirely. **The prototype is wired up; the real-world fusion path is an open research direction.**

## Known limitations

1. **Apple-Silicon Open3D 0.18 legacy RANSAC segfaults.** We replace it with a deterministic `n_hypotheses`-grid coarse-to-fine ICP. The replacement only covers yaw (Y-axis) rotations — pitch/roll variation is not in the grid. Acceptable for most upright-object multi-view cases but not arbitrary 6-DoF.
2. **Object-centric frames are not aligned across views.** As discussed above — the script will silently produce a wrong T when input frames are too dissimilar.
3. **Gaussian rotations are rotated correctly, but Gaussian *scales* / *opacities* are not blended.** Overlapping Gaussians stack as raw duplicates. A future iteration should either (a) prune overlaps via voxel-grid dedup or (b) average overlapping Gaussian parameters.
4. **`open3d` is heavy (~300 MB wheel)** and is not pinned in `requirements.txt`. Install on-demand: `pip install open3d`.

## Next steps

1. **Get a real multi-view dataset.** Even 2 photos of one object from 60° apart would be enough to test failure modes.
2. **Wire in DUSt3R or MASt3R** for image-side pose initialisation. Their pose estimates feed directly into `init_source_to_target` — no need for ICP global init at all.
3. **Voxel-dedup the merged Gaussians** to control output size.
4. **Quantitative re-rendering metric** (render each view from its original camera, measure PSNR vs the input image) for fusion quality scoring.
