# PORT_BLOCKERS.md

All Phase 2 items are now shipped. Original blockers are resolved; this
file documents what was done and what minor follow-ups (if any) remain.

## Phase 2 done (commits `c1ffe1b`, `b4c6803`, `c72b450`)

| Component                              | Module                                       | Parity test                          | Result |
|----------------------------------------|----------------------------------------------|--------------------------------------|--------|
| `unfold_nhwc` / `fold_nhwc`            | `propainter_mlx/sparse_transformer.py`       | `tests/test_sparse_transformer.py`   | max diff 0      |
| `SoftSplit` / `SoftComp`               | `propainter_mlx/sparse_transformer.py`       | `tests/test_sparse_transformer.py`   | 0 / 4e-5        |
| `FusionFeedForward`                    | `propainter_mlx/sparse_transformer.py`       | `tests/test_sparse_transformer.py`   | 7e-6            |
| `SparseWindowAttention`                | `propainter_mlx/sparse_transformer.py`       | `tests/test_sparse_transformer.py`   | (in block test) |
| `TemporalSparseTransformerBlock` (×8)  | `propainter_mlx/sparse_transformer.py`       | `tests/test_sparse_transformer.py`   | 2e-4            |
| `feat_prop_module` (learnable, C=128)  | `propainter_mlx/feat_prop.py`                | `tests/test_feat_prop.py`            | 1.4e-5          |
| `DeformableAlignment` (PP variant)     | `propainter_mlx/feat_prop.py`                | (in feat_prop test)                  | (passes)        |
| `flow_warp` + `fb_consistency_check`   | `propainter_mlx/feat_prop.py`                | `tests/test_feat_prop.py`            | 3e-6            |
| `InpaintGenerator` (top-level)         | `propainter_mlx/propainter.py`               | `tests/test_e2e.py`                  | 1e-5            |
| RAFT + RFC + main pipeline E2E         | (uses all of the above)                      | `tests/test_e2e_full_pipeline.py`    | PSNR 59.4 dB    |
| Video inpaint CLI                      | `scripts/inpaint_video_cli.py`               | manual (`test_outputs/bmx_inpaint.mp4`) | runs OK     |

## Memory / perf notes (still applicable)

* On M1 Max, 240×432 (480p-equivalent), 7 local frames + 3 ref frames,
  `InpaintGenerator.__call__` alone is **~129 ms / output-frame**.
* Full RAFT + RFC + img-prop + feat-prop + transformer pipeline on the
  same shape runs at **~450 ms / frame** end-to-end.
* Peak memory was well under 1 GB; no OOM observed.

## Known small follow-ups (not blockers)

1. **`SparseWindowAttention.__call__` per-batch loop** uses `mx.array.at[...].add(...)`
   for scatter, which works but is the slowest part of the forward pass.
   At B=1 this is a no-op; for B>1 inference (rare for video inpainting)
   it could be vectorised.

2. **`MaxPool2d` (`_max_pool2d_nhwc`) and bilinear interp** are
   implemented manually here because MLX's `nn.MaxPool2d` and
   `nn.functional.interpolate` semantics differ from PyTorch in edge
   cases (padding, `align_corners`). These local implementations
   exactly reproduce the PT semantics we need.

3. **fp16 inference**: the upstream `inference_propainter.py` supports
   `--fp16`. Our port stays in fp32; supporting fp16 in MLX is just a
   cast call but parity numbers would degrade.

4. **Reference-image-only chunking** in the CLI follows upstream's
   `neighbor_length`/`ref_stride`/`subvideo_length` semantics exactly.
   No deviation observed.

## Conv3d edge cases (already worked around)

MLX `Conv3d` since 0.31 supports asymmetric kernels / strides / dilation
but does not have `padding_mode='replicate'`. The RFC `downsample` Conv3d
uses replicate padding, which we work around manually in
`flow_completion._pad_replicate_dhw`. Still in place.

## Things NOT blockers (confirmed working)

* MLX `nn.Conv2d` / `Conv3d` with `groups>1` — confirmed working for
  Encoder (groups in {1, 2, 4, 8}) and `pool_layer` (groups=512).
* `mx.gather`-based bilinear sampling — verified bit-faithful to
  `F.grid_sample(align_corners=True, padding_mode='zeros')` in
  `propainter_mlx.raft.bilinear_sample_nhwc`.
* Modulated deformable conv 2D — verified against
  `torchvision.ops.deform_conv2d` to max-abs-diff 2e-6.
