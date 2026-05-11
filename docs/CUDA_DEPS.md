# SAM 3D Objects — CUDA-only ops in the inference path

Spec for the Metal-kernel agents (replacing CUDA libs with Metal kernels in MLX, not stubs).

Scope: every CUDA-only op reachable from `python demo.py` → `Inference(...)` → `inference(image, mask, seed)` → return `output["gs"]`. Output target is the Gaussian splat path (`decode_formats=["gaussian"]`). Mesh / texture / layout-postprocessing / video-render paths are out of scope but listed at the bottom for completeness.

Pipeline call chain (from `notebook/inference.py:107` and `sam3d_objects/pipeline/inference_pipeline_pointmap.py:385-509`):

```
inference(image, mask, seed)
└── pipeline.run(...)
    ├── compute_pointmap(image)               # depth_model = MoGe (timm/torch only, Mac OK)
    ├── preprocess_image(image, ss_preproc)
    ├── sample_sparse_structure(...)          # ss_generator = dense DiT (cold spconv: none)
    │   ├── ss_generator.reverse_fn(...)      # 25 ODE steps, dense backbone
    │   └── ss_decoder(latent)                # dense Conv3d, no spconv
    ├── pose_decoder(...)                     # pure torch
    ├── sample_slat(...)                      # slat_generator = sparse-DiT-with-conv-IO
    │   └── slat_generator.reverse_fn(...)    # 25 ODE steps, calls SparseConv3d each step
    ├── decode_slat(slat, formats=["gaussian"])
    │   └── slat_decoder_gs(slat)             # SparseLinear + sparse transformer (NO spconv)
    └── postprocess_slat_output(...)          # gaussian path: just unpacks; no rendering
```

CFG default doubles the per-step cost in both ODE samplers.

## Hot vs cold

- **Hot** = inside the per-ODE-step forward of either generator. Runs `inference_steps × (1 + cfg_active)` times = `~50` invocations per stage. Total `~100` invocations of the slat block stack per image at default `slat_inference_steps=25` with CFG.
- **Cold** = setup, decoding, postprocessing. Runs once per image.

---

## A. spconv (CUDA-only, no macOS wheel)

### A.1 `spconv.pytorch.SubMConv3d`  /  `spconv.pytorch.SparseConv3d`

- **File**: `sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_spconv.py:30, 40`
- **Wrapper**: `SparseConv3d.__init__` (lines 9-56). When `stride==1` and no padding it picks `SubMConv3d` (submanifold — output coords = input coords). Otherwise picks `SparseConv3d` (full sparse stride/pad). All call sites in this codebase use `kernel=3, stride=1` → submanifold; or `kernel=1, stride=1` → linear-like submanifold.
- **Forward**: `conv_spconv.py:58-89`. `new_data = self.conv(x.data)` is the CUDA call. After non-1 stride the output indices are re-sorted by batch (lines 64-73).
- **Signature**:
  - in: `SparseConvTensor` with `features [N, C_in]`, `indices [N, 4] int32 (batch, z, y, x)`, `spatial_shape [3]`, `batch_size`
  - weight: `[K, K, K, C_in, C_out]`
  - out: `SparseConvTensor` with `features [N', C_out]`, `indices [N', 4]`. For SubMConv3d N' = N and indices identical (submanifold rule).
- **Hot-path call sites** (per-ODE-step in slat_generator):
  - `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py:37` — `SparseResBlock3d.conv1` = `sp.SparseConv3d(channels, out_channels, 3)` (kernel 3, stride 1, sub-manifold)
  - `structured_latent_flow.py:39` — `SparseResBlock3d.conv2` = same
  - Each `SLatFlowModel` has `len(io_block_channels) * num_io_res_blocks` input + output `SparseResBlock3d`. With the default config (`patch_size=2`, `num_io_res_blocks=2`, single io-block-channels stage) that's **2 input + 2 output res-blocks × 2 conv each = 8 SparseConv3d forward calls per ODE step**.
  - Plus 1 `SparseLinear` skip-connection in each block when channel widths differ (cheap, `[N, C]` linear — already pure torch via `nn.Linear`).
- **Cold-path call sites** (mesh decoder, only if `decode_formats` includes `"mesh"`):
  - `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/decoder_mesh.py:47, 53, 65` — `SparseSubdivideBlock3d` uses `sp.SparseConv3d(... 3, indice_key=...)` and `sp.SparseConv3d(... 1, ...)`.
  - Mesh decoder is **not invoked** in the default Gaussian-only demo, but the slat_decoder_mesh.ckpt is loaded eagerly at pipeline init — so we still must instantiate the sparse stack even for the gaussian path. Loading parameters is OK without spconv if we replace the layer constructor.
- **Metal target**: `sparse_conv3d_submanifold(features, coords, weight, bias, kernel_size=3)` → `(features', coords)`. Output coords identical to input → no rebuild of hash table per call. Per-voxel work: gather 27 neighbors via coord hash, GEMM `27*C_in × C_out`. Since coords don't change across the ODE steps within one slat sample, **the neighbor table can be built once per inference and reused for all ~50 forward passes through the io blocks**. This is the single biggest hot-path optimization.

### A.2 `spconv.pytorch.SparseInverseConv3d`

- **File**: `sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_spconv.py:106`
- **Wrapper**: `SparseInverseConv3d` class (lines 92-139). Used to invert a previous strided sparse conv — recovers the pre-stride coord layout.
- **Hot-path call sites**: none in the slat generator forward we traced. Class exists in the codebase but no `sp.SparseInverseConv3d(` anywhere in `sam3d_objects/`.
- **Cold-path call sites**: none reachable from the gaussian-only demo path.
- **Metal target**: only port if a future config uses it. Treat as TODO.

### A.3 `spconv.pytorch.SparseConvTensor` (data class)

- **File**: `sam3d_objects/model/backbone/tdfy_dit/modules/sparse/basic.py:9, 85, 93`
- **Construction**: `basic.py:85` — `SparseTensorData(features.reshape(N, -1), coords, spatial_shape, batch_size)`. The wrapper `SparseTensor` in basic.py wraps either `torchsparse.SparseTensor` or `spconv.SparseConvTensor`; only the spconv branch ships in the inference config.
- **Hot/cold**: cold (just a container; no kernel). The features/coords members are accessed every step, but all reads/writes are pure tensor ops.
- **Metal target**: replace with a pure-MLX dataclass `(features: mlx.array [N, C], coords: mlx.array [N, 4] int32, spatial_shape: tuple[int,int,int], batch_size: int)`. No kernel. The submanifold-conv neighbor table belongs alongside this dataclass as a cache.

### A.4 spconv attribute access

- `conv_spconv.py:73` — constructs `spconv.SparseConvTensor(...)` with new sorted feats after a non-1-stride conv. With our submanifold-only hot path this branch never fires.
- `conv_spconv.py:121` — `data.replace_feature(...)` on the unsorted-cache spconv tensor (only used by `SparseInverseConv3d`).
- `conv_spconv.py:25-28` — `spconv.ConvAlgo.{Native, MaskImplicitGemm}` enum lookup. Cold, easy to skip.

---

## B. gsplat (CUDA-only, no macOS wheel)

### B.1 `gsplat.rasterization`

- **File of import**: `sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py:34`
- **Call site**: `gaussian_render.py:181-193`. Signature inside `render(...)`:
  ```python
  rasterization(
      means: [N, 3],           # gaussian centres (world)
      quats: [N, 4],           # rotations
      scales: [N, 3],          # log-space scales
      opacities: [N],
      colors: [N, K, 3] or None,   # SH coefs; K = (sh_degree+1)^2
      sh_degree: int,
      viewmats: [B, 4, 4],     # world->camera, row-vector convention
      Ks: [B, 3, 3],           # pixel-space intrinsics
      width, height: int,
      backgrounds: [B, 3],
  ) -> (render_colors [B, H, W, 3+], render_alphas [B, H, W, 1], meta)
  ```
- **Hot/cold**: **NOT in the default demo path.** `gaussian_render.render(...)` is called only by:
  - `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py:805` — texture-baking path. Demo passes `with_texture_baking=False`.
  - `sam3d_objects/model/backbone/tdfy_dit/utils/render_utils.py:94, 111, 148` — `render_frames` used by `notebook/inference.py:181 render_video(...)`. Not on demo.py path.
  - `sam3d_objects/pipeline/layout_post_optimization_utils.py:713, 1014, 1066, 1107` and `inference_utils.py:300, 373, 418` — layout post-opt loops, only run when `with_layout_postprocess=True`. Demo passes `False`.
  - **Demo never calls gsplat. Splatting is needed only for visualisation / training-time iteration.**
- **Metal target**: needed for the SAM-3D-Body-style visualization step and for layout post-opt. Implements full 3DGS: project each gaussian, sort by depth per tile, alpha-composite with per-pixel SH evaluation. Hot when used (run inside an inner-loop optimizer in layout post-opt). For Phase 0/1 we can ship without and write the .ply only.

### B.2 `diff_gaussian_rasterization` (Inria backend)

- **File**: `gaussian_render.py:24-32`. Wrapped in a `try/except ImportError` and only used when `backend="inria"`. The pipeline uses `backend="gsplat"` everywhere we traced.
- **Hot/cold**: dead in the inference path.
- **Metal target**: skip entirely.

---

## C. pytorch3d (CUDA-only on Linux, no macOS wheel from facebookresearch — would need source build with CUDA)

### C.1 Transforms (real math, ~10 functions)

These are torch-only on the *Linux* side too — pytorch3d packages them with optional CUDA kernels for batched ops, but every call we hit is pure-tensor math (matrix mul, quaternion arithmetic, look-at). No Metal kernel needed; just port to torch/MLX.

| Symbol | File:line called | Hot/cold |
|---|---|---|
| `pytorch3d.renderer.look_at_view_transform(eye=, at=, up=, device=)` | `inference_pipeline_pointmap.py:31` | Cold (once per `compute_pointmap`) |
| `pytorch3d.transforms.Transform3d` ctor + `.rotate(R) .inverse() .transform_points(p)` | `inference_pipeline_pointmap.py:274, 278, 301, 305` | Cold |
| `pytorch3d.transforms.quaternion_multiply` | `notebook/inference.py:272` (only inside `make_scene`, not in single-object demo path) | Cold |
| `pytorch3d.transforms.quaternion_invert` | `notebook/inference.py:273` (`make_scene` only) | Cold |
| `pytorch3d.transforms.quaternion_to_matrix` | `inference_utils.py:7`, `layout_post_optimization_utils.py:7, 17`, `pose_target.py:9` | Cold (pose_target.py runs inside SS preprocessor; the others run only when `with_layout_postprocess=True`) |
| `pytorch3d.transforms.matrix_to_quaternion` | `inference_utils.py:7`, `layout_post_optimization_utils.py:17`, `pose_target.py:9` | Cold |
| `pytorch3d.transforms.Transform3d` (used in pose_target.py / transforms_3d.py) | `pose_target.py:9`, `transforms_3d.py:6` | Cold (preprocessing) |
| `pytorch3d.transforms.{euler_angles_to_matrix, axis_angle_to_*}` | `transforms_3d.py:6` | Cold |

- **Metal target**: none. All replaceable by pure-torch / pure-MLX implementations. We already wrote a torch implementation at `mlx_port/stubs/pytorch3d/transforms/__init__.py` that covers `Transform3d`, `Translate`, `Scale`, `Rotate`, `RotateAxisAngle`, and the quaternion helpers. The MLX port should re-implement these in `mlx.array` (probably <300 LOC).

### C.2 Renderer + Structures (only used in mesh / vis paths)

| Symbol | File:line | Hot/cold |
|---|---|---|
| `pytorch3d.renderer.PerspectiveCameras` | `utils/visualization/scene_visualizer.py:5`, `notebook/mesh_alignment.py:16` | Out-of-scope (vis only) |
| `pytorch3d.renderer.MeshRasterizer` / `RasterizationSettings` | `notebook/mesh_alignment.py:16` | Out-of-scope |
| `pytorch3d.renderer.mesh.textures.TexturesVertex` | `utils/visualization/image_mesh.py:7`, `mesh_alignment.py:16` | Out-of-scope |
| `pytorch3d.structures.Meshes` | `inference_utils.py:6`, `layout_post_optimization_utils.py:6`, `image_mesh.py:6`, `mesh_alignment.py:15` | Cold; mesh / layout-postprocess only |
| `pytorch3d.structures.Pointclouds` | `scene_visualizer.py:6` | Vis only |
| `pytorch3d.vis.plotly_vis.*` | `utils/visualization/plotly/plot_scene.py:59` | Vis only |
| `pytorch3d.renderer.camera_utils.camera_to_eye_at_up` | `plot_scene.py:52` | Vis only |
| `pytorch3d.renderer.cameras.{CamerasBase, ...}` | `plot_scene.py:53` | Vis only |

- **Metal target**: not needed for Gaussian-only inference. The mesh-rasterizer would need a real Metal port if mesh rendering is added later (use `metal-rasterizer` crate or write our own; not Phase 0/1).

---

## D. kaolin (CUDA-only on Linux)

### D.1 `kaolin.utils.testing.check_tensor`

- **File**: `sam3d_objects/model/backbone/tdfy_dit/representations/mesh/flexicubes/flexicubes.py:18`
- **Call sites**: `flexicubes.py:60, 64, 67, 72, 76, 80` — six debug `check_tensor(..., throw=False)` calls inside `FlexiCubes.__call__`.
- **Hot/cold**: cold *and* dead in the gaussian-only demo (mesh decoder is loaded but not invoked).
- **Metal target**: none. Replace with a no-op (`def check_tensor(*a, **k): return None`) at import time. Trivial.

### D.2 `kaolin.visualize.IpyTurntableVisualizer`, `kaolin.render.camera.{Camera, ...}`

- **File**: `notebook/inference.py:25-26`. Imported at module level — `import inference` will fail without kaolin.
- **Hot/cold**: pure visualization, never called by `Inference.__call__`.
- **Action**: import is the only hard requirement. Either lazy-import these or stub the `kaolin` package at site-packages level so the `import` succeeds.

---

## E. nvdiffrast (CUDA-only)

### E.1 `nvdiffrast.torch as dr`

- **File**: `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py:555`
- **Guard**: only imported under `if rendering_engine == "nvdiffrast":`. The pipeline forces `rendering_engine="pytorch3d"` in `notebook/inference.py:88`, so this branch never executes.
- **Hot/cold**: dead in demo.
- **Metal target**: skip.

---

## F. diffoctreerast (CUDA-only, used by octree renderer)

### F.1 `diffoctreerast.{...}`

- **File**: `sam3d_objects/model/backbone/tdfy_dit/renderers/octree_renderer.py:58, 195`
- **Guard**: lazy import inside `render(...)`. Module already prints a warning and falls back if missing.
- **Hot/cold**: dead — `OctreeRenderer` is not invoked in the gaussian-only demo path; representation outputs are `Gaussian`, not `Strivec`.
- **Metal target**: skip.

---

## G. flash_attn / xformers (CUDA-only attention)

### G.1 attention backend selection

- **Files**: `sam3d_objects/model/backbone/tdfy_dit/modules/attention/full_attn.py:8-15`, `modules/sparse/attention/{serialized_attn,full_attn,windowed_attn}.py:9-13`
- **All four attention modules already include an `sdpa` branch** that uses `torch.nn.functional.scaled_dot_product_attention`. Setting `ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=sdpa` (the default when GPU name does not contain A100/H100/H200; see `inference_pipeline.py:11-22`) avoids both flash_attn and xformers entirely.
- **Hot/cold**: hot — runs inside every transformer block of every ODE step. Per-step volume:
  - ss_generator: full attention over `4096` tokens (16³ dense voxels) × `num_blocks` transformer blocks
  - slat_generator: cross attention against image conds + self-attention over `~N_voxel` sparse tokens × `num_blocks` blocks
  - slat_decoder_gs: windowed sparse attention over `~N_voxel * 8`(after subdivide) tokens × `num_blocks` blocks
- **Metal target**: this is the second-largest hot path. Equivalent of MLX's `mx.fast.scaled_dot_product_attention` (already supports causal/full self-attention and KV-packed forms). For the windowed/serialized sparse variants we need a custom Metal kernel that emulates `xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)` — i.e. a packed-batch block-diagonal mask SDPA. Without it we fall back to `masked_sdpa` (`modules/sparse/attention/masked_sdpa.py`), which builds an explicit `[1, M, M]` boolean mask — fine functionally, terrible for large M.

---

## Summary table — what needs Metal, what's just a torch-port

| Op | Lib | Hot? | Action |
|---|---|---|---|
| `SubMConv3d` (kernel=3, stride=1) | spconv | **HOT** (~8/step × 50 steps × 2 stages = ~800/image) | **Custom Metal kernel.** Build neighbor hash once per slat-gen call; reuse across all 50 ODE steps. |
| `SparseConv3d` (kernel≠1 or stride≠1) | spconv | dead in default config | Defer; only needed for mesh decoder. |
| `SparseInverseConv3d` | spconv | dead | Defer indefinitely. |
| `SparseConvTensor` container | spconv | n/a | Pure-MLX dataclass. |
| `gsplat.rasterization` | gsplat | dead in demo | **Custom Metal kernel** later, for video render & layout post-opt. Phase 1 ships .ply, no rendering. |
| `pytorch3d.transforms.*` (~8 fns) | pytorch3d | cold | Pure-torch / pure-MLX port, ~300 LOC. Already prototyped at `mlx_port/stubs/pytorch3d/transforms/`. |
| `pytorch3d.renderer.look_at_view_transform` | pytorch3d | cold | Pure-torch port (~30 LOC). |
| `pytorch3d.structures.Meshes`, `Pointclouds`, mesh renderer | pytorch3d | dead in demo | Skip for gaussian-only. |
| `kaolin.utils.testing.check_tensor` | kaolin | dead | No-op stub. |
| `kaolin.visualize.IpyTurntableVisualizer` etc. | kaolin | dead | Lazy-import or stub at notebook/inference.py module level. |
| `nvdiffrast.torch` | nvdiffrast | dead | Skip. |
| `diffoctreerast` | diffoctreerast | dead | Skip. |
| `flash_attn` / `xformers` | flash_attn/xformers | **HOT** | Already have sdpa fallback; for windowed sparse attn write a packed-batch block-diagonal-mask SDPA Metal kernel (or accept O(M²) mask matrix). |

## The two real Metal kernels

1. **Submanifold sparse 3D conv** (`SubMConv3d`, kernel 3³, stride 1, sub-manifold)
   - Inputs: `features [N, C_in]`, `coords [N, 4] int32 (b, z, y, x)`, `weight [3,3,3, C_in, C_out]`, `bias [C_out]`
   - Output: `features' [N, C_out]` (output coords == input coords)
   - Pre-pass: build a coord-hash → row-index lookup. Reuse across all calls in a single inference (coords never change in the slat io-block stack).
   - Per-voxel: 27 neighbor lookups, 27 × `C_in × C_out` MAC, accumulate.
   - Blocked by: nothing beyond a coord hash on Metal. ~150 LOC of MSL.

2. **Block-diagonal-mask SDPA** (for windowed / serialized sparse attention)
   - Inputs: `qkv [M, 3, H, C]` packed over windows of variable length, `seq_lens [B]` such that `sum(seq_lens) == M`
   - Output: `out [M, H, C]`
   - Equivalent to per-window unpadded SDPA (xformers `BlockDiagonalMask`). FlashAttention-style tiling, 1 block per window.
   - Fallback: explicit `[M, M]` mask + `mx.fast.scaled_dot_product_attention`. Acceptable for `M < 2048`.

Everything else is pure-torch / pure-MLX porting work, no kernels.
