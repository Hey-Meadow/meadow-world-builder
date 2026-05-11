# MOGE_RECON.md — MoGe pointmap conditioning recon

Status: investigation only, no code changes.
Date: 2026-05-08
Spec: `mlx_port/docs/SPEC_MOGE_RECON.md`

## TL;DR — recommended path

**Path (c) — synthetic fallback pointmap, NOT (a) and NOT (b).**

- `pointmap=None` does **not** short-circuit. The pipeline always builds a `pointmap_dict` and feeds it to `PointPatchEmbed`. There is no codepath in `InferencePipelinePointMap.run` that skips pointmap conditioning. So path (a) is wrong.
- The MoGe model is **DINOv2 ViT-L/14 + 5.6 M conv head = 309.9 M parameters / 1.26 GB fp32** (`Ruicheng/moge-vitl` on HF). MLX-porting it is technically feasible (the existing `lingbot-map/lingbot_map_mlx/dinov2_vit.py` already covers the backbone), but it is the second-largest model in the whole pipeline and reproducing the full `infer()` projection / focal-shift recovery is non-trivial. Not the right milestone for OBJ-INTEG.
- The clean abstraction is: the pipeline accepts a user-supplied `pointmap` (shape `(3, H, W)`, x/y/z in pytorch3d camera convention) and *only* falls through to MoGe if it is `None`. So a synthetic / classical-CV pointmap can be plumbed in without touching `sam3d_objects/`.

## Where MoGe is invoked

All paths below are absolute.

### Config wiring

`/Users/akaihuangm1/Desktop/github/sam-3d-objects/checkpoints/pipeline.yaml` lines 61–65:

```
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
  model:
    _target_: moge.model.v1.MoGeModel.from_pretrained
    pretrained_model_name_or_path: Ruicheng/moge-vitl
```

The `moge` package is allowed by the safety filter in `notebook/inference.py:38-40`:

```
WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]
```

### Wrapper (3 lines of work)

`/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/pipeline/depth_models/moge.py` (whole file, 11 LoC):

```python
class MoGe(DepthModel):
    def __call__(self, image):
        output = self.model.infer(image.to(self.device), force_projection=False)
        pointmaps = output["points"]
        output["pointmaps"] = pointmaps
        return output
```

i.e. the wrapper just calls `MoGeModel.infer(force_projection=False)` and renames `points` → `pointmaps`. `output` also carries `intrinsics`, `depth`, `mask`, `mask_prob`.

### Call sites in the pipeline

All inside `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py`:

- **Line 96** — `self.depth_model = depth_model` stored in `__init__`.
- **Line 148** — warmup loop: `pointmap_dict = recursive_clone(self.compute_pointmap(image))` (called 3× during `_warmup`).
- **Line 262–317** — `compute_pointmap(self, image, pointmap=None)`. The branch:
    - `pointmap is None` (line 268): `output = self.depth_model(loaded_image)` → `output["pointmaps"]`, then transform from MoGe camera convention to pytorch3d camera convention via `look_at_view_transform(eye=[0,0,-1], at=[0,0,0], up=[0,-1,0])` (line 31–41 + 273–278). Intrinsics taken from `output.get("intrinsics")`.
    - `pointmap is not None` (line 280): user-provided tensor is just moved to device and resized with `F.interpolate(..., mode="nearest")` to match image; **MoGe is never called**, and intrinsics are set to `None` so they get re-inferred by `infer_intrinsics_from_pointmap` (line 299–309) from the supplied pointmap.
- **Line 405** — main `run()`: `pointmap_dict = self.compute_pointmap(image, pointmap)` runs unconditionally on every inference call.
- **Line 414** — `ss_input_dict = self.preprocess_image(image, self.ss_preprocessor, pointmap=pointmap)`. The pointmap then flows into `PreProcessor._process_image_mask_pointmap_mess` (`sam3d_objects/data/dataset/tdfy/preprocessor.py:75-142`) which:
    1. Normalises with `ObjectCentricSSI` (median-shift + scene-scale) → `pointmap_scale`, `pointmap_shift`.
    2. Crops jointly with image+mask, resizes to 518×518.
    3. Stores `pointmap`, `rgb_pointmap`, `pointmap_scale`, `pointmap_shift`, `rgb_pointmap_scale`, `rgb_pointmap_shift` in the SS-input dict.
- **Line 218** — also stores `rgb_pointmap_unnorm` for layout post-optimization (only used when `with_layout_postprocess=True`; the `notebook/inference.py:116` default is `False`, so the unnorm pointmap is unused for OBJ-INTEG smoke runs).
- **Line 427–435** — `pointmap_scale` and `pointmap_shift` are passed into `pose_decoder` to rescale generated SS coords. So the pointmap also drives **scene-scale recovery**, not just per-token conditioning.

### Where the embedder consumes it

`PointPatchEmbed` lives at `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/model/backbone/dit/embedder/pointmap.py:12-238`. It is one of the embedders in `condition_embedders["ss_condition_embedder"].embedder_list` (see compile loop at `inference_pipeline_pointmap.py:108-117`). Input shape is `(B, 3, H, W)`; output is `(B, num_windows, D)` with D=768. Internally it nearest-down-samples to 256×256, projects (x,y,z)→768 with a single `nn.Linear(3, 768)`, runs one tiny window-level transformer block. It also has a `dropped_xyz_token` learnable embedding used during training when the whole pointmap is dropped (`apply_pointmap_dropout`, lines 108–153), and an `invalid_xyz_token` for NaN pixels (line 41, 181). Both tokens are present in the released checkpoint regardless of training-time dropout config.

## Does `pointmap=None` short-circuit?

**No.** Tracing the call chain:

1. `Inference.__call__(..., pointmap=None)` (`notebook/inference.py:101-120`) → `InferencePipelinePointMap.run(..., pointmap=None)`.
2. `run` line 405 unconditionally calls `compute_pointmap(image, pointmap)`.
3. `compute_pointmap` line 268: `if pointmap is None: output = self.depth_model(loaded_image)`. So `None` triggers MoGe rather than skipping.
4. `pointmap_dict["pointmap"]` is then handed back into `preprocess_image(..., pointmap=<populated tensor>)` at line 414.

Inside `preprocess_image` (line 173–220) there is a `if pointmap is not None and preprocessor.pointmap_transform != (None,)` guard (line 205, 214). In principle one could pass a preprocessor where `pointmap_transform == (None,)` to disable the pointmap path. But the released `pipeline.yaml` ss_preprocessor explicitly sets `pointmap_transform: torchvision.transforms.Compose(...)` (lines 52–59), so this guard is never hit.

Furthermore the SS-generator condition embedder list (in the checkpoint) contains `PointPatchEmbed`. If pointmap keys are missing from `ss_input_dict` the embedder will not get its input and the generator's condition tensor will be incomplete (`get_condition_input` in `inference_pipeline.py:628-642` builds positional args from `ss_condition_input_mapping=[]` (empty in the yaml), so all condition keys flow through `**condition_kwargs`, including `pointmap`, `rgb_pointmap`, `pointmap_scale`, `pointmap_shift`).

So: **pointmap is structurally required by the released checkpoint**. Path (a) is not viable as-is.

## MoGe model size & architecture

Verified by instantiating `moge.model.v1.MoGeModel(encoder='dinov2_vitl14', intermediate_layers=4, dim_proj=512, dim_upsample=[256,128,128])` and counting parameters (with the venv at `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv`):

| Component | Params | Notes |
|---|---:|---|
| DINOv2 ViT-L/14 backbone | 304.4 M | 24 blocks, dim 1024, 16 heads, patch 14, img 518 |
| Conv head (`Head` class, `moge/model/v1.py:60-141`) | 5.6 M | 4 projections @ 1024→512, 3 upsample blocks (with ResidualConvBlock + GroupNorm + UV concat), 2 output blocks (3-channel xyz + 1-channel mask) |
| **Total** | **309.9 M** | |

Weights file (HF API): `Ruicheng/moge-vitl/model.pt` is **1,256,823,446 bytes ≈ 1.26 GB** fp32. fp16 would be ~620 MB.

`infer()` does (`moge/model/v1.py:299-389`):
1. Resize input so `H'*W' ≈ num_tokens * 14²` (default ~2500 tokens at `resolution_level=9`).
2. ImageNet-mean/std normalize.
3. `backbone.get_intermediate_layers(image_14, 4, return_class_token=True)` — 4 intermediate ViT layers.
4. `head(features, image)` → `(points_xyz_pred, mask_pred)` at full input resolution.
5. `recover_focal_shift(points, mask>0.5)` — closed-form numpy-ish optimizer over focal & shift to make z-channel match a calibrated camera.
6. With `force_projection=False` (the value the wrapper uses), points are simply shifted in z by the recovered shift, no re-projection.

So the artefact PT consumes is *not* a metric depth map but a **camera-space xyz field**, where (x,y) follows MoGe's affine-invariant convention and z is depth-up-to-shift.

## Decision

**Recommend path (c): supply a synthetic / classical-CV pointmap via the existing `pointmap=` arg, do not port MoGe in this milestone.**

Reasoning:

1. **MoGe is 309.9 M params (1.26 GB)** — bigger than every other piece OBJ-INTEG is currently wiring up. Porting it expands the milestone scope by ≥1 week and risks blocking the integration smoke test. The DINOv2 ViT-L MLX port exists in `lingbot-map/lingbot_map_mlx/dinov2_vit.py`, but the head decoder (Head class, `moge/model/v1.py:60-141`) plus `recover_focal_shift` (`moge/utils/geometry_torch.py`) and `recover_focal_shift` numpy fallback have not been ported and would need new code.
2. **The pipeline already exposes the right hook**: passing a non-None `pointmap` makes `compute_pointmap` skip `self.depth_model` entirely (line 280–291). No edit to `sam3d_objects/` is required.
3. **The pointmap embedder is patch-window pooled and tolerates noise**: `PointPatchEmbed` resizes via `F.interpolate(..., mode="nearest")` to 256×256, then averages each 8×8 window down to one token (line 103-106, 188-225). It also has a learnt `dropped_xyz_token` and `invalid_xyz_token`, so the model has *seen* low-information pointmaps during training. This is the right place to inject a coarse fallback.
4. **OBJ-INTEG only needs to verify the MLX port runs end-to-end on a single image**. Image quality / scene-scale fidelity is a separate concern that can be improved later (e.g. with a tiny ported MoGe-S, or with metric-DepthAnything-MLX).

### Recommended fallback construction

A minimal viable fallback (built outside `sam3d_objects/`, e.g. in `mlx_port/scripts/` or wherever the integration entry point lives) is:

- Shape: `torch.Tensor` of shape `(3, H_img, W_img)` matching the alpha-cropped RGBA image. Will be resized by `compute_pointmap` line 286–290 anyway.
- Channel layout: pytorch3d camera-convention xyz (x right, y down, z forward → after the `look_at_view_transform([0,0,-1], at=[0,0,0], up=[0,-1,0])` rotation that the pipeline applies internally). For a fallback we can directly emit pytorch3d-convention by skipping the rotation.
- Construction (constant-z plane at unit depth, image-aligned x/y):
    - `u, v = meshgrid` normalized to `[-0.5, 0.5]` over (W, H).
    - `x = u`, `y = -v` (pytorch3d y-down vs image y-down — sign matches the rotation), `z = 1.0` everywhere inside the alpha mask.
    - `nan` outside the alpha mask, so `ObjectCentricSSI` ignores background and `invalid_xyz_token` is used for those tokens.
    - Stack to `(3, H, W)`.

This gives:
- correct image-aligned `(x,y)` coordinates (matches DINO patch grid),
- a single fronto-parallel-plane `z`, which means `recover_focal_shift` / `infer_intrinsics_from_pointmap` will produce a degenerate but finite intrinsics matrix (used only for layout post-optim, which is off by default in `notebook/inference.py:116`),
- `pointmap_scale` and `pointmap_shift` from `ObjectCentricSSI` will be defined (median over masked points),
- non-NaN values inside the object → `PointPatchEmbed` uses real point tokens; NaN outside → `invalid_xyz_token`.

Tradeoff vs MoGe: the model loses geometric prior on object depth structure. Empirically the SS generator is multimodal-conditioned on DINO image tokens too, so output should still be reasonable for smoke tests, but mesh quality will likely regress vs the MoGe-conditioned baseline. That is acceptable for milestone OBJ-INTEG and can be revisited.

### If quality regression is unacceptable later

In priority order:

1. **Reuse existing MLX DINOv2 port.** `lingbot-map/lingbot_map_mlx/dinov2_vit.py` already implements DINOv2 ViT (verified by user memory `reference_lingbot_map_mlx_ports.md`). Loading the `Ruicheng/moge-vitl` checkpoint into it requires re-keying the state_dict (`model.pt` keys are nested under `model['model']` per `moge/model/v1.py:230-235`).
2. **Port the MoGe `Head` decoder.** ~5.6 M params, plain conv + GroupNorm + ConvTranspose2d. Maybe ~250–400 LoC of MLX. Already implemented in `moge/model/v1.py:60-141`.
3. **Port `recover_focal_shift` + `_remap_points`.** Both small (one closed-form optimizer + scalar remap). Lives in `moge/utils/geometry_torch.py`.
4. **Skip the post-optim path** — `with_layout_postprocess=False` already in the integration entry point, so `infer_intrinsics_from_pointmap` does not need to be MLX-native (can stay torch / fall back to CPU).

Combined: porting full MoGe is feasible in MLX, ~1–2 k LoC, but explicitly out of OBJ-INTEG scope.

## Concrete answer for OBJ-INTEG

> The MLX integration agent should pass a synthetic pointmap (constant-z fronto-parallel plane, image-aligned x/y, NaN outside alpha mask) via the `pointmap=` kwarg of `InferencePipelinePointMap.run`. This bypasses MoGe entirely and exercises the same downstream code path as a real pointmap. **Do not** rely on `pointmap=None` — it triggers the full 309.9 M-param MoGe model. **Do not** port MoGe in this milestone.

## Files of record

- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline_pointmap.py` — main pipeline, lines 262-317 (`compute_pointmap`), 173-220 (`preprocess_image`), 385-509 (`run`).
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/pipeline/depth_models/moge.py` — MoGe wrapper.
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/pipeline/utils/pointmap.py` — `infer_intrinsics_from_pointmap`.
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/model/backbone/dit/embedder/pointmap.py` — `PointPatchEmbed`.
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/data/dataset/tdfy/preprocessor.py:75-146` — `_process_image_mask_pointmap_mess`.
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py:519-...` — `ObjectCentricSSI` normaliser.
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/notebook/inference.py:38-40, 101-120` — top-level `Inference` API + WHITELIST.
- `/Users/akaihuangm1/Desktop/github/sam-3d-objects/checkpoints/pipeline.yaml:61-65` — depth_model wiring.
- `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/lib/python3.11/site-packages/moge/model/v1.py` — MoGeModel definition.
- `/Users/akaihuangm1/Desktop/github/lingbot-map/lingbot_map_mlx/dinov2_vit.py` — existing MLX DINOv2 ViT port (reusable backbone).
