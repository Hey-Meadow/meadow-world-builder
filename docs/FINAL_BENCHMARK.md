# SAM 3D Objects MLX Final Benchmark

Date: 2026-05-10
Hardware: Apple M1 Max
Python: 3.11.12 (`sam-3d-body/.venv`)
Inference command (per object):

```bash
python mlx_port/infer_mlx.py \
    --image mlx_port/debug/input/<obj>/image.png \
    --mask  mlx_port/debug/input/<obj>/mask.png  \
    --use-moge --use-shortcut --dtype mixed --prune-outliers \
    --out /tmp/<obj>_final.ply
```

All three objects produced from the bundled debug inputs (`mlx_port/debug/input/{chair,table,plush}/`).
Reference PLYs are official Meta web demo outputs (`~/Downloads/object_0.ply`,
`objectb_0.ply`, `objectc_0.ply`).

---

## 1. Speed (M1 Max)

End-to-end wall-clock of `infer_mlx.py` (includes model load, MoGe, SS DiT, SLAT
DiT, GS decode, outlier prune, PLY save):

| Object | Wall total | preprocess | moge | ss_embed | ss_flow (4-step shortcut) | slat_embed | slat_flow (25-step CFG) | gs_decode | outlier_prune |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chair |  **86 s** | 0.02 s | 1.53 s | 2.05 s |  7.76 s | 1.83 s |  71.77 s | 0.78 s | 0.05 s |
| table |  **94 s** | 0.02 s | 1.56 s | 2.07 s |  8.72 s | 1.86 s |  79.00 s | 0.83 s | 0.07 s |
| plush | **122 s** | 0.03 s | 1.64 s | 2.41 s | 10.25 s | 1.99 s | 104.81 s | 1.08 s | 0.05 s |

(Wall total includes ~0.2 s weight load and PLY write; per-stage sums slightly
under the wall total.)

Average: **~100 s / object** (mean 100.7 s). Plush is slowest because its
SS produces the densest voxel grid (21 428 → pruned to 16 000), so SLAT-flow
runs over the maximum token count.

### Speedup vs baseline (~30 min / object on M1 Max, pre-optimization)

| Object | Baseline (estimate) | Now | Speedup |
|---|---:|---:|---:|
| chair | 1800 s | 86 s  | **20.9×** |
| table | 1800 s | 94 s  | **19.1×** |
| plush | 1800 s | 122 s | **14.7×** |

Mean speedup vs the 30-min baseline: **~18×**. The dominant remaining cost
(70–105 s) is `slat_flow` — 25 CFG steps × 2 forward passes through the SLAT
DiT — and is the next obvious target.

---

## 2. Quality vs Meta web demo

### Gaussian counts

| Object | MLX | Meta | Δ |
|---|---:|---:|---:|
| chair | 63 624 | 68 076 | −7 % |
| table | 64 000 | 64 380 | −0.6 % |
| plush | 64 000 | 51 340 | +25 % |

MLX always saturates near the 64 000 cap (16 000 voxels × 4 splats from
`gs_4`); Meta is moderately under-cap on plush (likely because the web demo
uses `gs_8` and prunes at decode time).

### Bounding box

| Object | MLX size (x, y, z) | Meta size (x, y, z) | Note |
|---|---|---|---|
| chair | [0.628, 0.703, 0.995] | [0.710, 0.706, 0.994] | MLX 12 % narrower in X |
| table | [0.970, 0.987, 0.727] | [0.953, 0.949, 0.741] | match within 4 % |
| plush | [0.881, 1.002, 0.806] | [0.638, 0.989, 0.546] | MLX 38 % wider in X / 47 % deeper in Z |

Table is the cleanest bbox match. Plush is the weakest — MLX produces a
larger, looser cloud (consistent with the visual blur seen in §4).

### Opacity / scale / quaternion

|  | MLX op (mean / med) | Meta op (mean / med) | MLX scale mean | Meta scale mean | MLX scale max | Meta scale max | \|q\| MLX | \|q\| Meta |
|---|---|---|---|---|---|---|---|---|
| chair | 0.943 / 0.981 | 0.897 / 0.973 | (0.006, 0.005, 0.004) | (0.003, 0.003, 0.003) | **clamped 0.010** | (0.013, 0.014, 0.019) | 1.0000 | 1.0000 |
| table | 0.971 / 0.992 | 0.976 / 0.991 | (0.007, 0.007, 0.003) | (0.004, 0.005, 0.003) | **clamped 0.010** | (0.015, 0.016, 0.021) | 1.0000 | 1.0000 |
| plush | 0.866 / 0.933 | 0.980 / 0.993 | (0.006, 0.008, 0.004) | (0.003, 0.003, 0.002) | **clamped 0.010** | (0.013, 0.017, 0.013) | 1.0000 | 1.0000 |

Observations:

1. **Quaternions are unit-norm on both sides** (the SLAT-POLISH normalize fix
   is in effect — every `|q|` reads exactly 1.0000).
2. **Scales are hard-clamped at 0.010** in MLX (`SLAT-POLISH` kernel cap,
   intentional). Meta lets occasional splats grow up to ~0.020. This makes MLX
   splats a bit smaller and helps prune obvious "stretchy" outliers, at the
   cost of slightly thicker clouds (mean scale is ~1.7–2× Meta on plush).
3. **Opacity distributions broadly match** for chair / table. Plush MLX has a
   noticeably lower mean opacity (0.866 vs 0.980) — fits the visible blur in
   the multi-view render.

### Outlier prune

`outlier_prune` step ran in 0.05–0.07 s per object and removed **0 voxels** in
all three cases (chair / table / plush were already clean at the chosen
defaults `radius=2.0, min_neighbors=3`). The prune is essentially a free
safety-net here; it would matter for noisier MoGe outputs.

---

## 3. Optimization stack (all enabled in this run)

| Optimization | Flag / env | Effect on this run |
|---|---|---|
| `gs_4` decoder swap | `SLAT_GS_VARIANT=gs_4` (default) | 4 splats / voxel → caps PLY at ~64 k Gaussians and removes 4 of the 8 decode heads |
| SLAT-POLISH GS fixes | always on in `decoder_mlx.py` | quaternion `qn` normalize, scale clamp at 9e-4 (in log-space → 0.010 in σ), kernel kept at 9e-4 |
| OUTLIER-PRUNE | `--prune-outliers` (default) | 0 voxels pruned this run; safety net for noisy MoGe |
| SHORTCUT-ENABLE | `--use-shortcut` | SS sampler: 25-step CFG-7 → **4-step distilled**; ~6.7× SS-flow speedup |
| BF16-MIXED | `--dtype mixed` | DiT blocks run in bf16 (matches PT `torch.autocast(bfloat16)`); ~1.4× DiT speedup |
| MoGe ViT-L on MLX | `--use-moge` | real depth via cached `moge_vitl.npz`, ~1.5 s |

---

## 4. Final visual outputs

PLYs:

- `/tmp/chair_final.ply` (4.33 MB, 63 624 Gaussians)
- `/tmp/table_final.ply` (4.35 MB, 64 000 Gaussians)
- `/tmp/plush_final.ply` (4.35 MB, 64 000 Gaussians)

Front-view PNGs:

- `/tmp/chair_final_preview/chair_final_preview.png`
- `/tmp/table_final_preview/table_final_preview.png`
- `/tmp/plush_final_preview/plush_final_preview.png`

Side-by-side vs Meta web demo:

- `/tmp/final_comparison/sidebyside.png` — single-view 3×2 grid
- `/tmp/final_comparison/sidebyside_4angles.png` — 4-azimuth 3×8 grid (recommended)

Per-object Meta renders for direct inspection:

- `/tmp/final_comparison/{chair,table,plush}_meta.png`
- `/tmp/final_comparison/{chair,table,plush}_mlx.png`

### Visual verdict (from `sidebyside_4angles.png`)

- **Chair**: MLX matches Meta closely on geometry (back / arms / seat all
  correct) and bbox. Colour is darker (the back/seat is rendered closer to
  black-brown vs Meta's red-orange leather). Likely a SLAT colour-prior
  difference, not a geometry issue.
- **Table**: Best of the three. Geometry, leg placement, table-top thickness
  all visually indistinguishable from Meta. Colour is slightly desaturated
  (paler yellow vs Meta's amber).
- **Plush**: Geometry is correct (chick body + beak + feet visible) but the
  cloud is fluffier / less defined than Meta's. Front-facing details (eyes,
  beak shape) are blurrier. Stats agree: bbox 38 % wider, mean scale 2× Meta,
  mean opacity 0.866 vs Meta's 0.980.

---

## 5. Remaining gaps (honest)

1. **`slat_flow` dominates** (70–105 s, 80–86 % of wall time). Still using the
   25-step CFG-7 baseline. SHORTCUT-ENABLE only accelerates the SS stage; the
   SLAT stage has no distilled shortcut yet. **Single highest-leverage next
   target.** A 4-step SLAT shortcut would push end-to-end to ~25–35 s / object.
2. **Scale clamp at 0.010** is conservative. Meta's web demo allows scales up
   to ~0.021. The clamp removes outliers but flattens fine geometry detail
   (most visible on the plush face). Worth A/B-ing a per-axis cap of
   `[0.014, 0.014, 0.014]` against the current `[0.010, 0.010, 0.010]`.
3. **Plush appearance gap.** Lower opacity mean + larger bbox + larger mean
   scale all point to under-fitted SLAT features. Possibly resolvable by
   higher SLAT CFG (currently default), or by raising the voxel cap (this run
   pruned 21 428 → 16 000).
4. **Colour cast** (chair darker, table paler). Consistent across re-runs;
   suggests the SLAT decoder colour-head is offset by a constant, not noise.
   Worth diff-checking the SH `f_dc_*` distribution against PT outputs on the
   same fixed noise.
5. **`gs_4` vs `gs_8`** — Meta likely uses `gs_8` (8 splats / voxel) and prunes
   at decode time. This explains why Meta plush has 51 340 splats (decode-time
   prune) while MLX always sits at the 16 000 × 4 = 64 000 cap. Switching MLX
   to `gs_8` + decode-time prune is another path to better quality at
   essentially identical inference cost.
6. **No `n_voxels_pruned` reporting** — the timing dict reports
   `n_voxels_pruned` as `0.00 s` (it's a count printed through the seconds
   format). Pure cosmetic; consider relabelling in the timing dict.

---

## Appendix: reproducible commands

```bash
# Run all three sequentially (~5 min total on M1 Max)
cd /Users/akaihuangm1/Desktop/github/sam-3d-objects
for n in chair table plush; do
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        mlx_port/infer_mlx.py \
        --image mlx_port/debug/input/$n/image.png \
        --mask  mlx_port/debug/input/$n/mask.png  \
        --use-moge --use-shortcut --dtype mixed --prune-outliers \
        --out /tmp/${n}_final.ply
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        mlx_port/scripts/render_ply.py /tmp/${n}_final.ply \
        --out /tmp/${n}_final_preview --no-mp4
done
```
