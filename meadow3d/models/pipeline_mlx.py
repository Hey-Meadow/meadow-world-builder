"""End-to-end SAM 3D Objects MLX inference pipeline.

Wires Phase 1 + Phase 2 modules into a single ``SAM3DObjectsPipeline`` class
that takes an RGBA image and produces a Gaussian-splat .ply.

Pipeline flow (mirrors PT ``inference_pipeline_pointmap.run``)::

    Stage 0   image preprocess  -> 518x518 RGB (NHWC, float)
                                + 256x256 dummy pointmap (no `moge` port)
    Stage 1   SS condition embedder -> (1, 7528, 1024) tokens
              (4 image trunks * 1370 + 2 pointmap trunks * 1024)
              MOTDiTBackbone via FlowMatching (CFG=7) -> shape latent (1, 4096, 8)
              SSDecoder reshape -> (1, 16, 16, 16, 8) -> occupancy (1, 64, 64, 64, 1)
              argwhere(>0) -> sparse coords (N, 4) at 64^3
    Stage 2   SLAT condition embedder -> (1, 5480, 1024) tokens (4 image trunks)
              Sparse U-Net + DiTBackbone + Sparse U-Net (24 transformer blocks)
              via FlowMatching (CFG=5) -> sparse latent (N, 8)
              denormalize with SLAT_MEAN/STD
    Stage 3   SLATDecoderGS(slat_feats, coords) -> Gaussian params
              save_gaussian_ply -> splat.ply

Pointmap policy (PT-faithful, single MoGe forward pass):
    PT runs MoGe ONCE on the original full-resolution image (see
    ``inference_pipeline_pointmap.compute_pointmap``). The resulting raw
    pointmap is then handed to ``ss_preprocessor._process_image_mask_pointmap_mess``
    which:
      1. Computes ObjectCentricSSI scale/shift from the FULL pointmap + FULL
         mask (BEFORE any crop). With ``allow_scale_and_shift_override=True``
         these get reused for both views, so the ``pointmap`` (cropped) and
         ``rgb_pointmap`` (full) views share normalization.
      2. Cropped view: joint transforms ``resize_all_to_same_size`` ->
         ``crop_around_mask_with_padding(box=1.2, pad=0.0)`` then individual
         ``pointmap_transform = pad_to_square_centered -> Resize(518, NEAREST)``.
      3. Full view: just the individual ``pointmap_transform`` on the
         normalized full pointmap (no crop).
    The MLX side now mirrors this exactly: one MoGe forward (or one dummy
    generation), ObjectCentricSSI on the full pointmap+mask, then
    transform-derived cropped/full views. When ``use_moge=False`` we substitute
    a synthetic constant-z dummy full pointmap; quality suffers but the shape
    contract matches PT's expectations bit-for-bit.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from meadow3d.kernels.sparse_subm_conv3d import clear_neighbor_cache
from meadow3d.models.decoder_mlx import (
    SLATDecoderGS,
    SSDecoder,
    save_gaussian_ply,
)
from meadow3d.models.dit_mlx import DiTBackbone, MOTDiTBackbone
from meadow3d.models.embedders_mlx import ConditionEmbedder
from meadow3d.models.latent_mapping_mlx import LatentMapping, OutputMapping
from meadow3d.models.sampler_mlx import FlowMatching
from meadow3d.models.sparse_blocks_mlx import (
    SparseInputBlocks,
    SparseOutputBlocks,
)


# ---------------------------------------------------------------------------
# SLAT mean/std (copied from sam3d_objects/pipeline/inference_utils.py).
# These denormalize the slat latent before the GS decoder.
# ---------------------------------------------------------------------------

SLAT_STD = mx.array(
    [
        2.37326008,
        2.13174402,
        2.2413953,
        2.30589401,
        2.1191894,
        1.8969511,
        2.41684989,
        2.08374642,
    ],
    dtype=mx.float32,
)
SLAT_MEAN = mx.array(
    [
        0.12211431,
        0.37204156,
        -1.26521907,
        -2.05276058,
        -3.10432536,
        -0.11294304,
        -0.85146744,
        0.45506954,
    ],
    dtype=mx.float32,
)


# ---------------------------------------------------------------------------
# Image preprocessing (numpy/PIL boundary)
# ---------------------------------------------------------------------------


def _crop_around_mask(rgba: np.ndarray, box_size_factor: float = 1.2,
                      padding_factor: float = 0.0) -> np.ndarray:
    """Reproduce PT ``crop_around_mask_with_padding`` using numpy/PIL.

    Mirrors ``sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py``:

      * ``compute_mask_bbox`` computes ``min_x/min_y/max_x/max_y`` (inclusive)
        over the alpha mask, then takes ``size = max(bbox_w, bbox_h, 2)``,
        scales by ``box_size_factor`` with ``int()`` truncation (NOT round),
        and re-centers around ``(cx +/- size//2, cy +/- size//2)``. The bbox
        returned by PT is always SQUARE.
      * The crop is taken with ``torchvision.transforms.functional.crop``
        which zero-pads when the bbox extends past the image edge.
      * ``crop_around_mask_with_padding`` then runs ``F.pad`` to make the
        result square (no-op when bbox is already square) and an optional
        10% extension ring (``padding_factor > 0``).

    PT yaml ``ss_preprocessor`` defaults: ``box_size_factor=1.2``,
    ``padding_factor=0.0`` (see ``checkpoints/pipeline.yaml`` lines 30-31, 87-88).

    rgba: (H, W, 4) uint8. Alpha channel is the object mask.
    Returns (H', W', 4) uint8 -- still RGBA.
    """
    H, W = rgba.shape[:2]
    mask = rgba[:, :, 3] > 0
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return rgba  # no mask -- return as-is

    # Inclusive min/max indices (PT's ``compute_mask_bbox`` uses
    # ``torch.nonzero(...).min()/.max()`` -- inclusive endpoints).
    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    # PT line 379-380: ``size = max(bbox_w, bbox_h, 2)``; ``size = int(size *
    # box_size_factor)`` -- truncation, not round. Final bbox is square.
    size = max(bbox_w, bbox_h, 2)
    size = int(size * box_size_factor)
    # Center is computed BEFORE truncation in PT (line 372-373: ``center_x =
    # (bbox[0] + bbox[2]) / 2``). Half-size uses ``size // 2`` which truncates
    # *before* the subtraction (PT lines 383-386).
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    half = size // 2
    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = int(center_x + half)
    y2 = int(center_y + half)

    # ``torchvision.transforms.functional.crop`` zero-pads when the bbox
    # extends past the image. Reproduce that here.
    y1c, y2c = max(0, y1), min(H, y2)
    x1c, x2c = max(0, x1), min(W, x2)
    crop = rgba[y1c:y2c, x1c:x2c]
    pad_top = y1c - y1   # > 0 only when y1 < 0
    pad_bot = y2 - y2c   # > 0 only when y2 > H
    pad_left = x1c - x1
    pad_right = x2 - x2c
    if pad_top or pad_bot or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((max(0, pad_top), max(0, pad_bot)),
             (max(0, pad_left), max(0, pad_right)),
             (0, 0)),
            mode="constant",
            constant_values=0,
        )
    Hc, Wc = crop.shape[:2]

    # PT lines 288-298: explicit pad-to-square AFTER the crop. For PT's square
    # bbox this is a no-op, but kept here for parity with the PT codepath.
    max_dim = max(Hc, Wc)
    pad_h = (max_dim - Hc) // 2
    pad_h_extra = (max_dim - Hc) - pad_h
    pad_w = (max_dim - Wc) // 2
    pad_w_extra = (max_dim - Wc) - pad_w
    if max_dim != Hc or max_dim != Wc:
        crop = np.pad(
            crop,
            ((pad_h, pad_h_extra), (pad_w, pad_w_extra), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # PT lines 308-316: optional 10% extension ring. Skipped under yaml
    # default (``padding_factor=0.0``); kept here for parity with PT's guard.
    if padding_factor > 0:
        ext = int(max_dim * padding_factor)
        crop = np.pad(
            crop,
            ((ext, ext), (ext, ext), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return crop


def _resize_rgba_pil(rgba: np.ndarray, size: int = 518) -> np.ndarray:
    """RGB -> bilinear (antialiased), mask -> nearest. Returns (size, size, 4) uint8.

    PT yaml uses ``torchvision.transforms.Resize(size=518)`` for the rgb branch
    (default ``InterpolationMode.BILINEAR`` with ``antialias=True``) and
    ``Resize(size=518, interpolation=0)`` (NEAREST) for the mask / pointmap
    branches (see ``checkpoints/pipeline.yaml`` lines 37-46, 72-83).

    PIL's ``Image.BILINEAR`` resample on uint8 tracks ``tv.Resize(BILINEAR,
    antialias=True)`` to within ~1/255 (verified on a 1024x1024 -> 518x518
    downsample: max abs delta < 0.004 in [0,1] space, mean < 0.0011). PIL
    BICUBIC, in contrast, drifts by up to 0.12 -- ~30x worse. The previous
    MLX port used ``Image.BICUBIC`` here; that is a real divergence from PT
    and is corrected below.

    NEAREST mask channel: PIL's ``Image.NEAREST`` and torchvision's
    ``InterpolationMode.NEAREST`` (interpolation=0 in the yaml) DO NOT MATCH
    on downscale. PIL uses round-to-nearest (i.e. ``round((i+0.5) * in/out -
    0.5)``) while torchvision NEAREST uses the legacy floor-snapping rule
    ``floor(i * in/out)`` (matches OpenCV INTER_NEAREST). On a 640->518 mask
    downscale, PIL vs torchvision disagree on ~85% of pixels. To match PT we
    therefore resample the mask channel with the torchvision rule directly,
    not via PIL. (``InterpolationMode.NEAREST_EXACT`` would match PIL but PT
    config uses the legacy NEAREST.)
    """
    rgb = rgba[:, :, :3]
    mask = rgba[:, :, 3]
    rgb_pil = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    mask_resized = _torchvision_nearest_resize_2d(mask, size, size)
    return np.concatenate(
        [np.asarray(rgb_pil), mask_resized[:, :, None]], axis=-1
    )


def _torchvision_nearest_resize_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Replicate ``torchvision.transforms.functional.resize(..., NEAREST)`` on a
    2D array. Used for the mask / pointmap branches whose PT config sets
    ``interpolation=0`` (legacy NEAREST). Verified against torchvision on a
    640->518 downscale: 0/268324 pixels differ.
    """
    in_h, in_w = arr.shape[:2]
    if in_h == out_h and in_w == out_w:
        return arr
    yi = (np.arange(out_h) * in_h / out_h).astype(np.int64)
    xi = (np.arange(out_w) * in_w / out_w).astype(np.int64)
    yi = np.clip(yi, 0, in_h - 1)
    xi = np.clip(xi, 0, in_w - 1)
    return arr[yi[:, None], xi[None, :]]


def preprocess_image_to_mlx(rgba_uint8: np.ndarray,
                            size: int = 518) -> Tuple[mx.array, mx.array]:
    """Cropped object preprocess: bbox crop -> pad-to-square -> resize.

    Mirrors PT ``ss_preprocessor`` joint transform
    (``crop_around_mask_with_padding`` w/ ``box_size_factor=1.2``,
    ``padding_factor=0.0``) followed by per-image transform
    (``pad_to_square_centered`` -> ``Resize(518)``).

    PT-faithful note: PT's SS preprocessor does NOT apply alpha-multiply
    (rembg) on the cropped RGB before feeding it to the DINO trunk. The
    ``image`` kwarg is just the cropped+resized rgb tensor (preserving the
    original rgb values at edge pixels where alpha is in (0,1)). The previous
    MLX port multiplied ``rgb *= mask`` here, which darkened edge pixels and
    cost ~3.3% cosine on the SS DINO image trunk (0.964 -> 0.996 once
    removed). Verified against PT path
    ``inference_pipeline_pointmap.preprocess_image`` (lines 187-203):
    ``rgb_image = rgba_image[:3]`` then joint transforms + img_transform with
    no mask multiplication anywhere.

    Returns (image_nhwc, mask_hw) as MLX float arrays in [0,1].
    """
    # PT yaml defaults: box_size_factor=1.2, padding_factor=0.0.
    rgba = _crop_around_mask(rgba_uint8, box_size_factor=1.2, padding_factor=0.0)
    rgba = _resize_rgba_pil(rgba, size)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    mask = (rgba[:, :, 3].astype(np.float32) / 255.0)
    return mx.array(rgb[None]), mx.array(mask)  # NHWC, HW


def _pad_to_square_centered(rgba: np.ndarray, fill: int = 0) -> np.ndarray:
    """Mirror PT ``img_processing.pad_to_square_centered`` for RGBA arrays.

    rgba: (H, W, 4) uint8.  Pads the smaller side equally on both sides so the
    result is square (max(H, W) on each side).
    """
    H, W = rgba.shape[:2]
    if H == W:
        return rgba
    diff = abs(H - W)
    pad1 = diff // 2
    pad2 = diff - pad1
    if H > W:
        pad = ((0, 0), (pad1, pad2), (0, 0))
    else:
        pad = ((pad1, pad2), (0, 0), (0, 0))
    return np.pad(rgba, pad, mode="constant", constant_values=fill)


def preprocess_image_full_to_mlx(
    rgba_uint8: np.ndarray, size: int = 518
) -> Tuple[mx.array, mx.array]:
    """PT-equivalent of ``rgb_image`` / ``rgb_image_mask`` (no object-bbox crop).

    PT ``inference_pipeline_pointmap.preprocess_image`` runs ``img_transform``
    (``pad_to_square_centered`` -> ``Resize(518)``) on the original RGB / mask
    pair to produce ``rgb_image`` and ``rgb_image_mask``. We mirror that here
    using PIL so the bicubic / nearest behaviour stays close to torchvision.

    Returns ``(rgb_full_nhwc, mask_full_hw)`` in [0, 1].
    """
    rgba = _pad_to_square_centered(rgba_uint8, fill=0)
    rgba = _resize_rgba_pil(rgba, size)
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    mask = rgba[:, :, 3].astype(np.float32) / 255.0
    # PT ``rgb_image`` comes from the rgb tensor BEFORE alpha-masking
    # (see ``inference_pipeline_pointmap.preprocess_image``: ``rgba_image[:3]``
    # is fed to ``preprocessor`` which only re-applies image / mask
    # transforms — no rembg multiplication on the full-image branch).
    return mx.array(rgb[None]), mx.array(mask)


def build_image_modality_inputs(
    rgba_uint8: np.ndarray, size: int = 518
) -> Dict[str, mx.array]:
    """Produce the four image-side kwargs consumed by the SS / SLAT embedders.

    PT ``embedder_list`` (see ``ss_generator.yaml``) requires per modality:
        image          : cropped object (rembg-masked RGB), NHWC
        rgb_image      : full image (pad-to-square + resize), NHWC
        mask           : cropped alpha (1-channel), NHWC
        rgb_image_mask : full alpha (1-channel),     NHWC

    ``image`` and ``mask`` come from the bbox-cropped path; ``rgb_image`` and
    ``rgb_image_mask`` come from the ``pad_to_square_centered`` path. Both are
    resized to ``size`` (518). Returned as a single dict so callers can splat
    it into ``embedder(**inputs)``.
    """
    image_nhwc, mask_cropped_hw = preprocess_image_to_mlx(rgba_uint8, size=size)
    rgb_full_nhwc, mask_full_hw = preprocess_image_full_to_mlx(rgba_uint8, size=size)

    # DINO consumes 1- or 3-channel NHWC; the mask branch keeps 1 channel and
    # is broadcast to RGB inside ``DinoViT._preprocess`` (matches PT
    # ``Dino._preprocess_input`` — ``if x.shape[1] == 1: repeat(...)``).
    mask_cropped_nhwc = mask_cropped_hw[None, :, :, None]
    mask_full_nhwc = mask_full_hw[None, :, :, None]

    return {
        "image": image_nhwc,
        "rgb_image": rgb_full_nhwc,
        "mask": mask_cropped_nhwc,
        "rgb_image_mask": mask_full_nhwc,
    }


# ---------------------------------------------------------------------------
# PT-faithful pointmap pipeline
#
# Mirrors ``sam3d_objects/data/dataset/tdfy/preprocessor.py`` +
# ``img_and_mask_transforms.py`` +
# ``inference_pipeline_pointmap.compute_pointmap``.
# ---------------------------------------------------------------------------


def _object_centric_ssi_full(
    pm_chw: np.ndarray, mask_hw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ObjectCentricSSI scale+shift on the FULL pointmap + FULL mask.

    Mirrors PT ``ObjectCentricSSI.normalize`` with the YAML defaults
    (``use_scene_scale=True``, ``scale_factor=1.0``, no clip). Returns the
    normalized pointmap plus (scale, shift) as 3-vectors. The same scale/shift
    are reused for the rgb_pointmap branch (PT sets
    ``allow_scale_and_shift_override=True`` so the pre-cropped scale/shift
    propagate directly).
    """
    H, W = pm_chw.shape[1:]
    flat = pm_chw.reshape(3, -1)  # (3, H*W)
    mask_bool = mask_hw.reshape(-1) > 0.5
    mask_pts = flat[:, mask_bool]  # (3, N_in_mask)

    if mask_pts.size == 0 or not np.isfinite(mask_pts).any():
        # Degenerate: no valid in-mask pixels. PT logs a warning and uses
        # scale=1, shift=0. Match that.
        scale = np.ones(3, dtype=np.float32)
        shift = np.zeros(3, dtype=np.float32)
        return pm_chw.copy(), scale, shift

    # shift = nanmedian over (x, y, z) channels of mask points.
    shift = np.array(
        [
            np.nanmedian(mask_pts[0]) if np.isfinite(mask_pts[0]).any() else 0.0,
            np.nanmedian(mask_pts[1]) if np.isfinite(mask_pts[1]).any() else 0.0,
            np.nanmedian(mask_pts[2]) if np.isfinite(mask_pts[2]).any() else 0.0,
        ],
        dtype=np.float32,
    )

    # scene scale: max over xyz of |center| at each pixel, then nanmedian over
    # ALL pixels (not just mask -- PT's ``self.use_scene_scale=True`` path
    # uses the full pointmap_flat). Equivalent to lines 562-565 of
    # img_and_mask_transforms.py.
    centered = flat - shift[:, None]
    with np.errstate(invalid="ignore"):
        max_dims = np.nanmax(np.abs(centered), axis=0)  # (H*W,)
    finite = max_dims[np.isfinite(max_dims)]
    if finite.size > 0:
        scene_scale = float(np.median(finite))
    else:
        scene_scale = 1.0
    scene_scale = max(scene_scale, 1e-6)
    scale = np.array([scene_scale, scene_scale, scene_scale], dtype=np.float32)

    # Apply normalization: (x - shift) / scale.
    pm_norm = (pm_chw - shift[:, None, None]) / scale[:, None, None]
    return pm_norm.astype(np.float32), scale, shift


def _pad_to_square_pm(pm_chw: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Mirror ``img_processing.pad_to_square_centered`` for a (3, H, W) pointmap.

    PT's ``pointmap_transform`` Compose calls ``pad_to_square_centered`` with
    no ``pointmap=`` kwarg, so it pads with ``value=0`` (NOT NaN). Matches
    line 128 of img_processing.py.
    """
    _, H, W = pm_chw.shape
    if H == W:
        return pm_chw
    diff = abs(H - W)
    pad1 = diff // 2
    pad2 = diff - pad1
    if H > W:
        pad = ((0, 0), (0, 0), (pad1, pad2))
    else:
        pad = ((0, 0), (pad1, pad2), (0, 0))
    return np.pad(pm_chw, pad, mode="constant", constant_values=fill_value)


def _resize_pm_nearest(pm_chw: np.ndarray, target_size: int) -> np.ndarray:
    """Per-channel NEAREST-neighbor resize matching ``Resize(518, interpolation=0)``.

    Uses PIL with mode='F' so float32 NaN handling is preserved in
    pass-through (NEAREST simply copies pixel values).
    """
    _, H, W = pm_chw.shape
    if H == target_size and W == target_size:
        return pm_chw
    out = np.empty((3, target_size, target_size), dtype=np.float32)
    for c in range(3):
        # ``Image.fromarray`` with mode='F' preserves float32 (incl. NaN).
        # PIL ``resize(..., Image.NEAREST)`` then nearest-samples.
        ch_pil = Image.fromarray(pm_chw[c].astype(np.float32), mode="F").resize(
            (target_size, target_size), Image.NEAREST
        )
        out[c] = np.asarray(ch_pil, dtype=np.float32)
    return out


def _crop_pointmap_with_mask(
    pm_chw: np.ndarray,
    mask_hw: np.ndarray,
    box_size_factor: float = 1.2,
    padding_factor: float = 0.0,
) -> np.ndarray:
    """Mirror ``crop_around_mask_with_padding`` for the pointmap branch.

    Returns the cropped + (NaN-padded to square + optional ring) pointmap.
    Matches ``img_and_mask_transforms.py`` lines 262-332. The pointmap path
    uses NaN as the pad fill value (line 305).
    """
    H, W = mask_hw.shape
    mask_bool = mask_hw > 0.5
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return pm_chw  # no mask -> passthrough

    min_y, max_y = int(ys.min()), int(ys.max())
    min_x, max_x = int(xs.min()), int(xs.max())
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    size = max(bbox_w, bbox_h, 2)
    size = int(size * box_size_factor)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    half = size // 2
    x1 = int(center_x - half)
    y1 = int(center_y - half)
    x2 = int(center_x + half)
    y2 = int(center_y + half)

    # Step 1: torchvision-style crop with zero-pad (the pointmap branch in PT
    # uses the SAME ``torchvision.transforms.functional.crop`` -- which pads
    # with 0 when the crop extends past edges -- BEFORE the explicit NaN-pad
    # to square. So we mirror that.) See line 284-286.
    y1c, y2c = max(0, y1), min(H, y2)
    x1c, x2c = max(0, x1), min(W, x2)
    crop = pm_chw[:, y1c:y2c, x1c:x2c]
    pad_top = max(0, y1c - y1)
    pad_bot = max(0, y2 - y2c)
    pad_left = max(0, x1c - x1)
    pad_right = max(0, x2 - x2c)
    if pad_top or pad_bot or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,  # tv.crop pads with 0 on out-of-bounds
        )

    # Step 2: explicit pad-to-square with NaN (matches lines 300-306).
    Cn, Hc, Wc = crop.shape
    max_dim = max(Hc, Wc)
    pad_h = (max_dim - Hc) // 2
    pad_h_extra = (max_dim - Hc) - pad_h
    pad_w = (max_dim - Wc) // 2
    pad_w_extra = (max_dim - Wc) - pad_w
    if max_dim != Hc or max_dim != Wc:
        crop = np.pad(
            crop,
            ((0, 0), (pad_h, pad_h_extra), (pad_w, pad_w_extra)),
            mode="constant",
            constant_values=np.nan,
        )

    # Step 3: optional 10% extension ring (NaN). Default padding_factor=0.0
    # in pipeline.yaml so this is a no-op for SAM 3D Objects.
    if padding_factor > 0:
        ext = int(max_dim * padding_factor)
        crop = np.pad(
            crop,
            ((0, 0), (ext, ext), (ext, ext)),
            mode="constant",
            constant_values=np.nan,
        )

    return crop.astype(np.float32)


def _apply_pt_pointmap_pipeline(
    pm_full_chw: np.ndarray,
    mask_full_hw: np.ndarray,
    target_size: int,
) -> Tuple[mx.array, mx.array]:
    """Run PT's pointmap preprocessor on a (3, H, W) full pointmap + full mask.

    Returns ``(pointmap_cropped_mlx, pointmap_full_mlx)`` -- both
    (1, 3, target_size, target_size) MLX float32 arrays. Mirrors
    ``preprocessor._process_image_mask_pointmap_mess`` exactly:

      1. SSI-normalize the full pointmap with the FULL mask (returns shared
         scale/shift).
      2. Cropped path: crop_around_mask_with_padding -> pad_to_square_centered
         (no-op once square) -> Resize(target_size, NEAREST).
      3. Full path: pad_to_square_centered (zero-pad) -> Resize(target_size,
         NEAREST). The full pointmap re-uses the SAME scale/shift via
         ``allow_scale_and_shift_override=True``.

    No camera-convention rotation is applied here -- callers that produce a
    fresh MoGe pointmap must already have applied PT's pytorch3d-look_at flip
    (xy-negation). When loading the precomputed ``moge_pointmap.npz`` the
    saved values are already PT-equivalent (no flip needed; PT skips the
    rotation when an external pointmap is supplied -- see
    ``inference_pipeline_pointmap.compute_pointmap`` lines 280-291).
    """
    pm_norm, _scale, _shift = _object_centric_ssi_full(pm_full_chw, mask_full_hw)

    # Cropped path.
    pm_cropped = _crop_pointmap_with_mask(
        pm_norm, mask_full_hw, box_size_factor=1.2, padding_factor=0.0
    )
    # Already square after _crop_pointmap_with_mask, so pad_to_square is a
    # no-op. Apply Resize(target_size, NEAREST).
    pm_cropped = _pad_to_square_pm(pm_cropped, fill_value=0.0)
    pm_cropped = _resize_pm_nearest(pm_cropped, target_size)

    # Full path: just pad-to-square + resize.
    pm_full = _pad_to_square_pm(pm_norm, fill_value=0.0)
    pm_full = _resize_pm_nearest(pm_full, target_size)

    return mx.array(pm_cropped[None]), mx.array(pm_full[None])


def make_moge_pointmap(
    rgba_uint8: np.ndarray,
    moge_model=None,
    pm_size: int = 518,
    apply_pytorch3d_flip: bool = True,
    precomputed_full_pm: Optional[np.ndarray] = None,
) -> Tuple[mx.array, mx.array]:
    """Run MoGe ONCE on the full image; derive cropped+full views via PT transforms.

    PT runs MoGe a single time on the original full-resolution image (see
    ``inference_pipeline_pointmap.compute_pointmap``) and feeds the result
    through ``ss_preprocessor`` which derives both the ``pointmap`` and
    ``rgb_pointmap`` views via deterministic transforms. The pre-fix MLX port
    ran MoGe twice (once on cropped, once on full), which produced
    independent pointmaps with different SSI normalizations -- a real
    divergence from PT and the leading suspect for the chair noise-box
    output. This implementation now mirrors PT exactly.

    Args:
        rgba_uint8: (H, W, 4) uint8. Alpha channel encodes the object mask.
        moge_model: pre-loaded ``MoGeModel`` (lazy-loaded when None).
        pm_size: side of the output pointmap. PT yaml sets ``Resize(518)``.
            ``PointPatchEmbed`` resizes internally to 256 either way.
        apply_pytorch3d_flip: if True (default), apply the look_at_view
            xy-negation that PT's ``compute_pointmap`` adds when MoGe runs
            fresh. Skip when feeding a precomputed pointmap (PT skips the
            rotation in that branch).
        precomputed_full_pm: optional (3, H, W) raw pointmap to use instead of
            running MoGe. Used by the dump scripts to consume
            ``moge_pointmap.npz`` (the same input PT consumes), eliminating
            MLX-vs-PT MoGe-forward variance from the comparison.

    Returns:
        pointmap_cropped: (1, 3, pm_size, pm_size) cropped+SSI-normalized.
        pointmap_full:    (1, 3, pm_size, pm_size) full+SSI-normalized.
    """
    H, W = rgba_uint8.shape[:2]
    mask_full = (rgba_uint8[:, :, 3].astype(np.float32) / 255.0)

    if precomputed_full_pm is not None:
        # ``moge_pointmap.npz`` already contains NaN outside the mask + raw
        # MoGe coords (no pytorch3d flip). PT consumes it as-is.
        pm_full = precomputed_full_pm.astype(np.float32)
        if pm_full.shape[1:] != (H, W):
            # Defensive resize; should not trigger when prepare_input.py
            # produced the bundle on the same image.
            pm_full = _resize_pm_nearest(pm_full, max(H, W))
    else:
        # Fresh MoGe forward on the FULL original image (no pad / crop /
        # resize -- MoGe handles internal resize itself).
        from meadow3d.models.moge_mlx import get_or_load_moge

        if moge_model is None:
            moge_model = get_or_load_moge()
        rgb_norm = rgba_uint8[:, :, :3].astype(np.float32) / 255.0
        out = moge_model(mx.array(rgb_norm[None]), apply_mask=False)
        points = np.asarray(out["points"])[0].astype(np.float32)  # (3, H', W')
        mask_pred = np.asarray(out["mask"])[0, 0] > 0.0  # (H', W') bool

        # Defensive shape correction: MoGe's internal Upsample uses a float
        # scale_factor (target/source); for some image dims floating-point
        # precision drops the result by 1 px (e.g. 820 -> 819 because
        # 714 * (820/714) = 819.99999... and int() truncates). The trim
        # ``x[:, :h, :w]`` inside ``_resize_bilinear_nhwc`` only truncates,
        # never pads, so the MoGe output can end up (H, W-1) or (H-1, W).
        # Reproduces on the table fixture (792, 820) but not on the chair
        # (640, 640) where scales are exact integers.
        # Edge-replicate to (H, W) for ``points`` and zero-pad for ``mask_pred``
        # (treat the missing border as outside-mask) so downstream broadcast
        # works.
        if points.shape[1:] != (H, W) or mask_pred.shape != (H, W):
            ph, pw = points.shape[1:]
            if ph != H or pw != W:
                pad_h = H - ph
                pad_w = W - pw
                if pad_h < 0 or pad_w < 0:
                    # Output is LARGER than expected (shouldn't happen with
                    # int() truncation, but handle defensively by cropping).
                    points = points[:, :H, :W]
                    mask_pred = mask_pred[:H, :W]
                    ph, pw = points.shape[1:]
                    pad_h = H - ph
                    pad_w = W - pw
                if pad_h > 0 or pad_w > 0:
                    points = np.pad(
                        points,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="edge",
                    )
                    mask_pred = np.pad(
                        mask_pred,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=False,
                    )

        mask_alpha = mask_full > 0.5
        valid = mask_pred & mask_alpha

        if apply_pytorch3d_flip:
            # PT ``compute_pointmap`` lines 273-278: rotate via
            # look_at_view_transform([0,0,-1], at=[0,0,0], up=[0,-1,0]) which
            # flips x and y signs while leaving z unchanged.
            points = points.copy()
            points[0] = -points[0]
            points[1] = -points[1]

        # NaN outside mask (PT's load_rgb -> compute_pointmap chain leaves
        # MoGe-invalid pixels finite in the new branch but ``rembg`` zeros
        # them; the SSI normalizer treats NaN/inf consistently. We use NaN
        # for explicit "ignore me" semantics that match
        # ``prepare_input.run_moge``).
        points[:, ~valid] = np.nan
        pm_full = points

    return _apply_pt_pointmap_pipeline(pm_full, mask_full, pm_size)


def make_dummy_pointmap(
    rgba_uint8: np.ndarray,
    pm_size: int = 518,
) -> Tuple[mx.array, mx.array]:
    """Synthesize a constant-z dummy full pointmap, then derive cropped+full views.

    Same shape contract as :func:`make_moge_pointmap` but no MoGe forward.
    Used for fast smoke-tests when the 309M-param MoGe checkpoint is
    unavailable; final geometry is degraded but the SS / SLAT code paths
    exercise unchanged.

    Args:
        rgba_uint8: (H, W, 4) uint8 (alpha = mask).
        pm_size:    target pointmap side after preprocess.
    """
    H, W = rgba_uint8.shape[:2]
    mask_full = rgba_uint8[:, :, 3].astype(np.float32) / 255.0

    # Dummy full pointmap: linear xy grid spanning [-1, 1] over the original
    # image, constant z=1, NaN outside the mask. Geometry-wise the values
    # don't match MoGe but the contract (mask -> finite, outside -> NaN)
    # holds.
    yy = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :]
    pm_full = np.stack(
        [
            np.broadcast_to(xx, (H, W)).copy(),
            np.broadcast_to(yy, (H, W)).copy(),
            np.ones((H, W), dtype=np.float32),
        ],
        axis=0,
    )
    pm_full[:, mask_full <= 0.5] = np.nan
    return _apply_pt_pointmap_pipeline(pm_full, mask_full, pm_size)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SAM3DObjectsPipeline:
    """End-to-end MLX inference pipeline. Single-object, single-batch."""

    def __init__(
        self,
        ss_embedder: ConditionEmbedder,
        ss_backbone: MOTDiTBackbone,
        ss_latent_mapping: LatentMapping,
        ss_output_mapping: OutputMapping,
        ss_decoder: SSDecoder,
        slat_embedder: ConditionEmbedder,
        slat_backbone: DiTBackbone,
        slat_input_layer: nn.Linear,
        slat_out_layer: nn.Linear,
        slat_input_blocks: SparseInputBlocks,
        slat_output_blocks: SparseOutputBlocks,
        slat_decoder_gs: SLATDecoderGS,
        ss_t_embed_weight: Dict[str, mx.array],
    ):
        self.ss_embedder = ss_embedder
        self.ss_backbone = ss_backbone
        self.ss_latent_mapping = ss_latent_mapping
        self.ss_output_mapping = ss_output_mapping
        self.ss_decoder = ss_decoder
        self.slat_embedder = slat_embedder
        self.slat_backbone = slat_backbone
        self.slat_input_layer = slat_input_layer
        self.slat_out_layer = slat_out_layer
        self.slat_input_blocks = slat_input_blocks
        self.slat_output_blocks = slat_output_blocks
        self.slat_decoder_gs = slat_decoder_gs
        self._ss_t_embed_weight = ss_t_embed_weight

    # ---- loading -----------------------------------------------------------
    @classmethod
    def from_npz_dir(
        cls,
        npz_dir: str = "meadow3d/weights/sam3d_objects",
        dtype: str = "mixed",
    ) -> "SAM3DObjectsPipeline":
        """Load pipeline from npz weights.

        Args:
            npz_dir: directory containing the converted MLX npz files.
            dtype: precision policy for the SS / SLAT DiT backbones.
                ``"fp32"`` keeps everything in fp32 (legacy, slowest).
                ``"bf16"`` is an alias for ``"mixed"``: per-block weights
                bf16, embedders / latent_mapping / decoders / samplers /
                AdaLN cond paths fp32. ``"mixed"`` (default) mirrors PT's
                ``torch.autocast(dtype=bfloat16)`` wrapper around the
                generators (see
                ``sam3d_objects/pipeline/inference_pipeline.py:71, 674``).
                Checkpoints are bf16-trained, so this matches PT semantics
                while keeping numerically sensitive paths in fp32.
        """
        npz_dir_p = Path(npz_dir)

        dtype_norm = (dtype or "mixed").lower()
        if dtype_norm == "fp32":
            block_dtype = mx.float32
        elif dtype_norm in ("bf16", "bfloat16", "mixed"):
            block_dtype = mx.bfloat16
        elif dtype_norm in ("fp16", "float16", "half"):
            block_dtype = mx.float16
        else:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Expected one of: fp32 / fp16 / bf16 / mixed."
            )

        # ---- embedders ----
        ss_embedder = ConditionEmbedder.from_npz(str(npz_dir_p / "ss_embedder.npz"))
        slat_embedder = ConditionEmbedder.from_npz(
            str(npz_dir_p / "slat_embedder.npz")
        )

        # ---- ss_flow: MOTDiTBackbone + LatentMapping + OutputMapping ----
        ss_backbone = MOTDiTBackbone.from_npz(
            str(npz_dir_p / "ss_flow.npz"),
            depth=24,
            channels=1024,
            num_heads=16,
            ctx_channels=1024,
            mlp_ratio=4.0,
            qk_rms_norm=True,
            qk_rms_norm_cross=False,
            share_mod=False,
            has_d_embedder=True,
            block_dtype=block_dtype,
        )
        ss_weights = mx.load(str(npz_dir_p / "ss_flow.npz"))
        ss_latent_mapping = LatentMapping.from_npz(
            ss_weights, prefix="reverse_fn.backbone.latent_mapping.",
            model_channels=1024,
        )
        ss_output_mapping = OutputMapping.from_npz(
            ss_weights, prefix="reverse_fn.backbone.latent_mapping.",
            model_channels=1024,
        )

        # ---- ss_decoder ----
        ss_decoder = SSDecoder.from_npz(str(npz_dir_p / "ss_decoder.npz"))

        # ---- slat_flow: DiT + sparse U-Net wrappers ----
        slat_backbone = DiTBackbone.from_npz(
            str(npz_dir_p / "slat_flow.npz"),
            depth=24,
            channels=1024,
            num_heads=16,
            ctx_channels=1024,
            mlp_ratio=4.0,
            qk_rms_norm=True,
            qk_rms_norm_cross=False,
            share_mod=False,
            block_dtype=block_dtype,
        )

        slat_weights = mx.load(str(npz_dir_p / "slat_flow.npz"))
        # SparseLinear at slat input/output (NOT the same as latent_mapping).
        slat_input_layer = nn.Linear(8, 128, bias=True)
        slat_input_layer.weight = slat_weights[
            "reverse_fn.backbone.input_layer.weight"
        ]
        slat_input_layer.bias = slat_weights[
            "reverse_fn.backbone.input_layer.bias"
        ]
        slat_out_layer = nn.Linear(128, 8, bias=True)
        slat_out_layer.weight = slat_weights[
            "reverse_fn.backbone.out_layer.weight"
        ]
        slat_out_layer.bias = slat_weights["reverse_fn.backbone.out_layer.bias"]

        slat_input_blocks = SparseInputBlocks.from_npz(
            slat_weights, prefix="reverse_fn.backbone.input_blocks.", emb_channels=1024
        )
        slat_output_blocks = SparseOutputBlocks.from_npz(
            slat_weights, prefix="reverse_fn.backbone.out_blocks.", emb_channels=1024
        )

        # ---- slat_decoder_gs ----
        # gs vs gs_4: 32 vs 4 Gaussians per voxel. Meta web demo uses gs_4
        # (cleaner, 8x fewer Gaussians, smaller .ply). Default to gs_4.
        import os as _os
        gs_variant = _os.environ.get("SLAT_GS_VARIANT", "gs_4")
        slat_decoder_gs = SLATDecoderGS.from_npz(
            str(npz_dir_p / f"slat_decoder_{gs_variant}.npz")
        )

        return cls(
            ss_embedder=ss_embedder,
            ss_backbone=ss_backbone,
            ss_latent_mapping=ss_latent_mapping,
            ss_output_mapping=ss_output_mapping,
            ss_decoder=ss_decoder,
            slat_embedder=slat_embedder,
            slat_backbone=slat_backbone,
            slat_input_layer=slat_input_layer,
            slat_out_layer=slat_out_layer,
            slat_input_blocks=slat_input_blocks,
            slat_output_blocks=slat_output_blocks,
            slat_decoder_gs=slat_decoder_gs,
            ss_t_embed_weight={},
        )

    # ---- forward -----------------------------------------------------------

    def __call__(
        self,
        rgba_uint8: np.ndarray,
        seed: int = 42,
        ss_steps: int = 25,
        slat_steps: int = 25,
        ss_cfg: float = 7.0,
        slat_cfg: float = 5.0,
        timing: Optional[Dict[str, float]] = None,
        use_moge: bool = False,
        prune_outliers: bool = True,
        outlier_radius: float = 2.0,
        outlier_min_neighbors: int = 3,
        use_shortcut: bool = False,
        shortcut_steps: int = 4,
        input_size: int = 518,
    ) -> dict:
        """Run end-to-end. Input: HxWx4 uint8 RGBA (mask in alpha).

        Args:
            use_moge: if True, use real MoGe pointmap (309M-param ViT-L).
                If False, fall back to the synthetic constant-z dummy
                pointmap that produces noisy output but exercises the same
                code path.
        """
        if timing is None:
            timing = {}

        # ---- Stage 0: image preprocess ----
        # ``input_size`` controls the side length fed to the DINO trunk and the
        # PointPatchEmbed pointmap branch. Default 518 matches PT (canonical
        # DINOv2 grid 37x37). pos_embed is bilinearly interpolated for any
        # other patch grid inside ``DinoViT._interpolate_pos_embed``.
        # Must be a multiple of 14 (DINO patch size). 1024 -> snapped to 1022
        # (73x73 grid); 1036 -> 74x74; etc. We round DOWN to the nearest
        # multiple so user-friendly inputs like ``--input-size 1024`` work.
        ps = 14
        if input_size % ps != 0:
            snapped = (input_size // ps) * ps
            print(
                f"[run] input_size={input_size} not a multiple of {ps}; "
                f"snapping to {snapped}."
            )
            input_size = snapped
        t0 = time.time()
        image_inputs = build_image_modality_inputs(rgba_uint8, size=input_size)
        image_nhwc = image_inputs["image"]
        rgb_full_nhwc = image_inputs["rgb_image"]
        mask_cropped_hw = image_inputs["mask"][0, :, :, 0]
        mask_full_hw = image_inputs["rgb_image_mask"][0, :, :, 0]
        timing["preprocess"] = time.time() - t0

        # ---- Stage 0.5: pointmap generation (MoGe or dummy) ----
        # PT runs MoGe ONCE on the full original image, then derives the
        # ``pointmap`` (cropped) and ``rgb_pointmap`` (full) views via
        # transforms inside ``ss_preprocessor``. We mirror that here.
        t0 = time.time()
        if use_moge:
            pointmap, rgb_pointmap = make_moge_pointmap(
                rgba_uint8, pm_size=input_size,
            )
        else:
            pointmap, rgb_pointmap = make_dummy_pointmap(
                rgba_uint8, pm_size=input_size,
            )
        mx.eval(image_nhwc, mask_cropped_hw, pointmap, rgb_pointmap)
        timing["moge" if use_moge else "dummy_pointmap"] = time.time() - t0

        # ---- Stage 1: SS condition embedder ----
        # PT ``ss_generator.yaml`` requires 6 kwargs (image / rgb_image / mask
        # / rgb_image_mask / pointmap / rgb_pointmap). Anything missing would
        # surface as a ``KeyError`` inside ``EmbedderFuser``.
        t0 = time.time()
        ss_cond_tokens = self.ss_embedder(
            **image_inputs, pointmap=pointmap, rgb_pointmap=rgb_pointmap,
        )
        mx.eval(ss_cond_tokens)
        timing["ss_embed"] = time.time() - t0

        # ---- Stage 1: SS sample (MOT flow) ----
        # If `use_shortcut`, run the distilled shortcut model (4-step, no CFG)
        # instead of the 25-step CFG-7 baseline. Both paths share the same
        # backbone weights; the only difference is whether the d_embedder is
        # fed (d=1/N) or zeroed out.
        t0 = time.time()
        if use_shortcut:
            coords = self._sample_ss(
                ss_cond_tokens, seed=seed, num_steps=shortcut_steps,
                cfg=0.0, use_shortcut=True,
            )
        else:
            coords = self._sample_ss(
                ss_cond_tokens, seed=seed, num_steps=ss_steps, cfg=ss_cfg,
            )
        timing["ss_flow"] = time.time() - t0
        timing["ss_mode"] = (
            f"shortcut-{shortcut_steps}step" if use_shortcut else f"cfg-{ss_steps}step"
        )

        if coords.shape[0] == 0:
            raise RuntimeError(
                "SS decoder produced 0 occupied voxels -- model likely diverged."
            )

        # Memory cap: SLAT self-attention is full O(N^2). Without flash-attn we
        # can't fit > ~8k tokens at 32^3 (post-downsample). Surface-prune +
        # random subsample. With a real pointmap (MoGe) the SS flow naturally
        # produces a clean ~10-30k coord set; with the dummy pointmap fallback
        # used here it can blow up to 250k+ -- prune aggressively to keep the
        # pipeline memory-feasible.
        n_voxels_pre_prune = int(coords.shape[0])
        timing["n_voxels_pre_prune"] = n_voxels_pre_prune
        import os as _os
        max_voxels = int(_os.environ.get("MLX_SS_MAX_VOXELS", "16000"))
        if n_voxels_pre_prune > max_voxels:
            print(f"[ss] {n_voxels_pre_prune} voxels -> pruning to <={max_voxels}")
            coords = _surface_prune(coords, max_voxels=max_voxels, seed=seed)
        else:
            print(f"[ss] {n_voxels_pre_prune} voxels (no prune)")

        # ---- Stage 2: SLAT condition embedder ----
        # PT ``slat_generator.yaml`` only consumes the four image-side kwargs
        # (no pointmap branch): image / rgb_image / mask / rgb_image_mask.
        t0 = time.time()
        slat_cond_tokens = self.slat_embedder(**image_inputs)
        mx.eval(slat_cond_tokens)
        timing["slat_embed"] = time.time() - t0

        # ---- Stage 2: SLAT sample ----
        t0 = time.time()
        slat_feats = self._sample_slat(
            slat_cond_tokens, coords, seed=seed, num_steps=slat_steps, cfg=slat_cfg
        )
        timing["slat_flow"] = time.time() - t0

        # ---- Denormalize ----
        slat_feats = slat_feats * SLAT_STD + SLAT_MEAN

        # ---- Stage 3: GS decoder ----
        t0 = time.time()
        gs = self.slat_decoder_gs(slat_feats, coords)
        mx.eval(*gs.values())
        timing["gs_decode"] = time.time() - t0

        # ---- Stage 3.5: outlier prune (drop floating voxels) ----
        # Local-density check on voxel coords. Voxels whose neighborhood
        # contains fewer than ``outlier_min_neighbors`` points within
        # ``outlier_radius`` (in voxel-grid units) are treated as floating
        # noise and all G Gaussians per voxel are removed. Cheap (KDTree on
        # <=16k points) and avoids the high-opacity speckle visible in the
        # SuperSplat viewer above/around the main mass.
        if prune_outliers:
            t0 = time.time()
            coords, gs, n_pruned = _prune_outlier_gaussians(
                coords, gs,
                density_radius=outlier_radius,
                min_neighbors=outlier_min_neighbors,
            )
            mx.eval(coords, *gs.values())
            timing["outlier_prune"] = time.time() - t0
            timing["n_voxels_pruned"] = n_pruned
            print(
                f"[run] pruned {n_pruned} outlier voxels "
                f"(radius={outlier_radius}, min_neighbors={outlier_min_neighbors})"
            )

        return {
            "voxels": coords,
            "gs_params": gs,
            "n_voxels": int(coords.shape[0]),
            "timing": timing,
        }

    # ---- SS stage helper ---------------------------------------------------
    def _sample_ss(
        self,
        cond_tokens: mx.array,
        seed: int,
        num_steps: int,
        cfg: float,
        use_shortcut: bool = False,
    ) -> mx.array:
        """Run SS flow with all 5 modalities, take `shape` -> SSDecoder -> coords."""
        # Per-modality init shapes (B=1):
        #   shape (1, 4096, 8), 6drotation_normalized (1, 1, 6),
        #   scale (1, 1, 3), translation (1, 1, 3), translation_scale (1, 1, 1)
        modality_shapes = {
            "shape": (1, 4096, 8),
            "6drotation_normalized": (1, 1, 6),
            "scale": (1, 1, 3),
            "translation": (1, 1, 3),
            "translation_scale": (1, 1, 1),
        }
        # latent_share_transformer: merge {6drotation_normalized, translation, scale,
        # translation_scale} into one transformer-key "6drotation_normalized" of length 4.
        # Order MUST match PT ss_generator.yaml latent_share_transformer list
        # (translation BEFORE scale).
        share_groups = {
            "6drotation_normalized": [
                "6drotation_normalized", "translation", "scale", "translation_scale",
            ]
        }

        rng = np.random.default_rng(seed)
        latents = {}
        for name, shp in modality_shapes.items():
            latents[name] = mx.array(rng.standard_normal(shp).astype(np.float32))

        # Backbone wrapper for FlowMatching.
        # FlowMatching expects backbone_fn(x, t, cond) where x is a single tensor
        # (we'll flatten/unflatten the latent dict into a single concatenated array
        #  and reuse the (B, N_total, max_in_dim) trick? -- simpler: keep dict and
        # write our own integrator).
        if use_shortcut:
            return self._mot_shortcut_sample(
                latents=latents,
                cond_tokens=cond_tokens,
                share_groups=share_groups,
                num_steps=num_steps,
            )
        return self._mot_euler_sample(
            latents=latents,
            cond_tokens=cond_tokens,
            share_groups=share_groups,
            num_steps=num_steps,
            cfg=cfg,
        )

    def _mot_euler_sample(
        self,
        latents: Dict[str, mx.array],
        cond_tokens: mx.array,
        share_groups: Dict[str, List[str]],
        num_steps: int,
        cfg: float,
    ) -> mx.array:
        """Custom Euler integrator for the multi-modality SS flow.

        Returns coords (N, 4) int32 from SSDecoder argwhere(>0).
        """
        time_scale = 1000.0
        # Build linear t schedule.
        t_seq_np = np.linspace(0.0, 1.0, num_steps + 1).astype(np.float32)
        # Sub-step Euler.
        x = dict(latents)
        for i in range(num_steps):
            t0 = float(t_seq_np[i])
            t1 = float(t_seq_np[i + 1])
            dt = t1 - t0
            t_in = mx.array([t0 * time_scale], dtype=mx.float32)
            v_cond = self._mot_backbone_velocity(
                x, t_in, cond_tokens, share_groups
            )
            if cfg > 0.0:
                # CFG: also evaluate at zero conditioning.
                uncond_tokens = mx.zeros_like(cond_tokens)
                v_uncond = self._mot_backbone_velocity(
                    x, t_in, uncond_tokens, share_groups
                )
                v = {
                    k: (1.0 + cfg) * v_cond[k] - cfg * v_uncond[k]
                    for k in v_cond
                }
            else:
                v = v_cond
            x = {k: x[k] + v[k] * dt for k in x}
            mx.eval(*x.values())

        # x["shape"] is (1, 4096, 8) denoised. Reshape -> (1, 16, 16, 16, 8) for SSDecoder.
        shape_lat = x["shape"]
        shape_cube = shape_lat.reshape(1, 16, 16, 16, 8)
        occ = self.ss_decoder(shape_cube)  # (1, 64, 64, 64, 1)
        mx.eval(occ)
        # argwhere (occ > 0)
        occ_np = np.asarray(occ)[0, :, :, :, 0]  # (64, 64, 64)
        zs, ys, xs = np.nonzero(occ_np > 0)
        if zs.size == 0:
            return mx.array(np.zeros((0, 4), dtype=np.int32))
        b = np.zeros_like(zs, dtype=np.int32)
        coords = np.stack([b, zs.astype(np.int32), ys.astype(np.int32), xs.astype(np.int32)], axis=-1)
        return mx.array(coords)

    def _mot_shortcut_sample(
        self,
        latents: Dict[str, mx.array],
        cond_tokens: mx.array,
        share_groups: Dict[str, List[str]],
        num_steps: int = 4,
    ) -> mx.array:
        """Shortcut-model sampler for the SS MOT flow.

        Mirrors PT ``ShortCut.generate_iter`` (sam3d_objects/model/backbone/
        generator/shortcut/model.py). Differences vs ``_mot_euler_sample``:

        * ``num_steps`` is small (4 by default) instead of 25.
        * No CFG -- distilled cond+uncond is baked into the model. We only
          forward the conditional path once per step (~6x fewer backbone
          evals than 25-step CFG-7).
        * Each step feeds ``d = 1/num_steps`` (in raw [0,1] units, scaled by
          ``time_scale`` to match PT) into the d_embedder so the model knows
          which step size to consume.

        Returns coords (N, 4) int32, same as ``_mot_euler_sample``.
        """
        time_scale = 1000.0
        # Linear t schedule on [0, 1].
        t_seq_np = np.linspace(0.0, 1.0, num_steps + 1).astype(np.float32)
        # Constant step size in raw-time units (NOT time_scale-scaled here;
        # we scale below to match PT's `d * time_scale`).
        d_raw = 1.0 / float(num_steps)
        d_in = mx.array([d_raw * time_scale], dtype=mx.float32)

        x = dict(latents)
        for i in range(num_steps):
            t0 = float(t_seq_np[i])
            t1 = float(t_seq_np[i + 1])
            dt = t1 - t0
            t_in = mx.array([t0 * time_scale], dtype=mx.float32)
            v = self._mot_backbone_velocity(
                x, t_in, cond_tokens, share_groups, d=d_in
            )
            x = {k: x[k] + v[k] * dt for k in x}
            mx.eval(*x.values())

        # x["shape"] is (1, 4096, 8) denoised. Reshape -> (1, 16, 16, 16, 8)
        # for SSDecoder.
        shape_lat = x["shape"]
        shape_cube = shape_lat.reshape(1, 16, 16, 16, 8)
        occ = self.ss_decoder(shape_cube)  # (1, 64, 64, 64, 1)
        mx.eval(occ)
        occ_np = np.asarray(occ)[0, :, :, :, 0]  # (64, 64, 64)
        zs, ys, xs = np.nonzero(occ_np > 0)
        if zs.size == 0:
            return mx.array(np.zeros((0, 4), dtype=np.int32))
        b = np.zeros_like(zs, dtype=np.int32)
        coords = np.stack(
            [b, zs.astype(np.int32), ys.astype(np.int32), xs.astype(np.int32)],
            axis=-1,
        )
        return mx.array(coords)

    def _mot_backbone_velocity(
        self,
        latents: Dict[str, mx.array],
        t: mx.array,
        cond_tokens: mx.array,
        share_groups: Dict[str, List[str]],
        d: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        """Apply latent_mapping (in) -> MOTDiTBackbone -> latent_mapping (out).

        ``d`` is the shortcut step-size embedding input (matches PT
        ``ShortCut._generate_dynamics``). When ``d`` is None the d_embedder is
        bypassed and the model behaves like the un-distilled flow-matching net.
        """
        # 1. Per-modality input projection
        projected = {n: self.ss_latent_mapping.modalities[n].to_input(latents[n])
                     for n in latents}
        # 2. Merge share_groups (concat along token dim)
        merged: Dict[str, mx.array] = {}
        merged_into = set()
        for new_name, members in share_groups.items():
            tensors = [projected[m] for m in members]
            merged[new_name] = mx.concatenate(tensors, axis=1)
            merged_into.update(members)
        for n in projected:
            if n not in merged_into:
                merged[n] = projected[n]

        # 3. Run backbone (only routes 'shape' and '6drotation_normalized')
        out = self.ss_backbone(merged, t, cond_tokens, d=d)

        # 4. Split share_groups back
        split: Dict[str, mx.array] = {}
        seen = set()
        for new_name, members in share_groups.items():
            seen.add(new_name)
            cat = out[new_name]
            offset = 0
            for m in members:
                tlen = self.ss_latent_mapping.modalities[m].pos_emb.shape[0]
                split[m] = cat[:, offset:offset + tlen]
                offset += tlen
        for n in out:
            if n not in seen:
                split[n] = out[n]

        # 5. Per-modality out projection
        velocities = {n: self.ss_output_mapping.modalities[n].to_output(split[n])
                      for n in latents}
        return velocities

    # ---- SLAT stage --------------------------------------------------------
    def _sample_slat(
        self,
        cond_tokens: mx.array,
        coords: mx.array,
        seed: int,
        num_steps: int,
        cfg: float,
    ) -> mx.array:
        """Run SLAT flow at given sparse coords. Returns (N, 8) sparse latent."""
        # Clear submconv neighbor cache so we rebuild for THIS coord set.
        clear_neighbor_cache()
        N = int(coords.shape[0])
        rng = np.random.default_rng(seed + 1)
        x = mx.array(rng.standard_normal((N, 8)).astype(np.float32))

        # Pre-compute the downsampled coords + per-input bucket idx ONCE.
        # Coordinates don't change per ODE step -- only the latent feats do --
        # so this runs once and the SubMConv3d neighbor-table cache covers
        # all 25 sub-steps.
        coords_down, idx_up = _avg_pool_coords(coords, factor=2)
        mx.eval(coords_down, idx_up)

        # Time schedule.
        t_seq_np = np.linspace(0.0, 1.0, num_steps + 1).astype(np.float32)
        time_scale = 1000.0
        for i in range(num_steps):
            t0 = float(t_seq_np[i])
            t1 = float(t_seq_np[i + 1])
            dt = t1 - t0
            t_in = mx.array([t0 * time_scale], dtype=mx.float32)
            v_cond = self._slat_backbone_velocity(
                x, coords, coords_down, idx_up, t_in, cond_tokens
            )
            if cfg > 0.0:
                uncond = mx.zeros_like(cond_tokens)
                v_uncond = self._slat_backbone_velocity(
                    x, coords, coords_down, idx_up, t_in, uncond
                )
                v = (1.0 + cfg) * v_cond - cfg * v_uncond
            else:
                v = v_cond
            x = x + v * dt
            mx.eval(x)
        return x

    def _slat_backbone_velocity(
        self,
        x: mx.array,
        coords: mx.array,
        coords_down: mx.array,
        idx_up: mx.array,
        t: mx.array,
        cond_tokens: mx.array,
    ) -> mx.array:
        """Sparse U-Net + DiT + reverse U-Net for slat_flow.

        x: (N, 8) sparse latent feats at `coords` (64^3).
        Returns velocity (N, 8).
        """
        # 1. input_layer (Linear 8 -> 128) on N feats
        h = self.slat_input_layer(x)  # (N, 128)

        # 2. t_emb -> 1024
        t_emb = self.slat_backbone.t_embedder(t)  # (1, 1024)

        # 3. input_blocks
        # block 0: no downsample, in=128 out=128, on coords
        h = self.slat_input_blocks.blocks[0](h, coords, t_emb)
        skip0 = h  # (N, 128)
        # block 1: downsample then conv (in=128 out=1024)
        # Do the downsample on h+coords -> coords_down, ds_feats (N_down, 128).
        h_down = _scatter_mean(h, idx_up, coords_down.shape[0])
        h_down = self.slat_input_blocks.blocks[1](h_down, coords_down, t_emb)
        skip1 = h_down  # (N_down, 1024)

        # 4. APE on coords_down
        h_down = h_down + self._slat_ape(coords_down[:, 1:])

        # 5. DiT (24 transformer blocks). Need (B, N_down, 1024) and cross-attn cond.
        h_dit = h_down[None]  # (1, N_down, 1024)
        h_dit = self.slat_backbone(h_dit, t, cond_tokens)
        # NOTE: the backbone above re-runs t_embedder -- harmless, t_emb stays the same.
        h_down_post = h_dit[0]  # (N_down, 1024)

        # 6. out_blocks (skip threading mirrors PT line for line):
        #   out_blocks[0] has upsample=True. PT: cat(h, skip1) at N_down (1024+1024
        #   -> 2048ch), then SparseUpsample expands N_down -> N, then conv1
        #   (2048 -> 128) etc.
        h_cat = mx.concatenate([h_down_post, skip1], axis=-1)   # (N_down, 2048)
        out_blk0 = self.slat_output_blocks.blocks[0]
        # Step A: upsample (gather using idx_up: each voxel reads its bucket's feat).
        h_up_cat = h_cat[idx_up]                                 # (N, 2048)
        # Step B: norm1 + silu + conv1
        h2 = out_blk0.norm1(h_up_cat)
        h2 = nn.silu(h2)
        h2 = out_blk0.conv1(h2, coords)  # (N, 128)
        # Step C: norm2 + AdaLN
        emb_out = out_blk0.emb_proj(nn.silu(t_emb))
        scale, shift = mx.split(emb_out, 2, axis=-1)
        if scale.ndim == 2 and scale.shape[0] == 1:
            scale = scale[0]
            shift = shift[0]
        mu = mx.mean(h2, axis=-1, keepdims=True)
        var = mx.mean((h2 - mu) * (h2 - mu), axis=-1, keepdims=True)
        h2 = (h2 - mu) * mx.rsqrt(var + 1e-6)
        h2 = h2 * (1.0 + scale) + shift
        h2 = nn.silu(h2)
        h2 = out_blk0.conv2(h2, coords)  # (N, 128)
        # Step D: skip
        if out_blk0.skip_connection is not None:
            h2 = h2 + out_blk0.skip_connection(h_up_cat)
        else:
            h2 = h2 + h_up_cat
        # h2 now (N, 128) at coords.

        # out_blocks[1]: cat with skip0 (N, 128) -> 256 ch -> conv (256->128)
        # No upsample.
        h_cat2 = mx.concatenate([h2, skip0], axis=-1)  # (N, 256)
        out_blk1 = self.slat_output_blocks.blocks[1]
        h3 = out_blk1.norm1(h_cat2)
        h3 = nn.silu(h3)
        h3 = out_blk1.conv1(h3, coords)
        emb_out2 = out_blk1.emb_proj(nn.silu(t_emb))
        scale2, shift2 = mx.split(emb_out2, 2, axis=-1)
        if scale2.ndim == 2 and scale2.shape[0] == 1:
            scale2 = scale2[0]
            shift2 = shift2[0]
        mu = mx.mean(h3, axis=-1, keepdims=True)
        var = mx.mean((h3 - mu) * (h3 - mu), axis=-1, keepdims=True)
        h3 = (h3 - mu) * mx.rsqrt(var + 1e-6)
        h3 = h3 * (1.0 + scale2) + shift2
        h3 = nn.silu(h3)
        h3 = out_blk1.conv2(h3, coords)
        if out_blk1.skip_connection is not None:
            h3 = h3 + out_blk1.skip_connection(h_cat2)
        else:
            h3 = h3 + h_cat2

        # Final LayerNorm + out_layer (Linear 128 -> 8). PT uses no-affine F.layer_norm.
        mu = mx.mean(h3, axis=-1, keepdims=True)
        var = mx.mean((h3 - mu) * (h3 - mu), axis=-1, keepdims=True)
        h3 = (h3 - mu) * mx.rsqrt(var + 1e-6)
        out = self.slat_out_layer(h3)
        return out

    def _slat_ape(self, xyz_coords: mx.array) -> mx.array:
        """Sin/cos APE on (x, y, z) -- mirrors PT AbsolutePositionEmbedder."""
        channels = 1024
        in_channels = 3
        freq_dim = channels // in_channels // 2
        freqs = mx.arange(freq_dim, dtype=mx.float32) / freq_dim
        freqs = 1.0 / (10000.0 ** freqs)
        flat = xyz_coords.astype(mx.float32).reshape(-1)
        out = flat[:, None] * freqs[None, :]
        out = mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)
        out = out.reshape(xyz_coords.shape[0], in_channels * 2 * freq_dim)
        if out.shape[1] < channels:
            pad = mx.zeros((out.shape[0], channels - out.shape[1]))
            out = mx.concatenate([out, pad], axis=-1)
        return out


# ---------------------------------------------------------------------------
# Sparse helpers (used by slat sample)
# ---------------------------------------------------------------------------


def _surface_prune(coords: mx.array, max_voxels: int = 16000, seed: int = 0) -> mx.array:
    """Keep at most `max_voxels` voxels, preferring surface (low-neighbor-count) ones.

    Mirrors PT `prune_sparse_structure` (boundary detection via 3x3x3 conv) plus
    a random subsample if still over-budget. Pure numpy hot path; coords is
    typically O(100k) at first-pass with the dummy pointmap, dropping to O(10k)
    by the time we leave this function.
    """
    coords_np = np.asarray(coords).astype(np.int64)
    N = coords_np.shape[0]
    # Build occupancy grid for one batch (we run single-image inference).
    if N == 0:
        return coords
    b = coords_np[:, 0]
    if not np.all(b == b[0]):
        # multi-batch path -- just random subsample
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(N), size=min(max_voxels, N), replace=False)
        return mx.array(coords_np[idx].astype(np.int32))
    # Build dense occupancy at 64^3 (sparse structure resolution).
    R = int(coords_np[:, 1:].max()) + 1
    R = max(R, 64)
    occ = np.zeros((R, R, R), dtype=np.uint8)
    occ[coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]] = 1
    # 3x3x3 box convolution via cumulative trick (or use scipy if available).
    try:
        from scipy.ndimage import uniform_filter
        counts_grid = uniform_filter(occ.astype(np.int32) * 27, size=3, mode="constant")
    except Exception:
        # Manual 3x3x3 box sum via pad + slide.
        padded = np.pad(occ, 1, mode="constant").astype(np.int32)
        counts_grid = np.zeros_like(occ, dtype=np.int32)
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    counts_grid += padded[dz:dz + R, dy:dy + R, dx:dx + R]
    counts = counts_grid[coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]]
    is_surface = counts < 27
    surface_idx = np.nonzero(is_surface)[0]
    if surface_idx.size > max_voxels:
        rng = np.random.default_rng(seed)
        surface_idx = rng.choice(surface_idx, size=max_voxels, replace=False)
    if surface_idx.size == 0:
        # Fallback: random subsample from all coords.
        rng = np.random.default_rng(seed)
        surface_idx = rng.choice(np.arange(N), size=min(max_voxels, N), replace=False)
    return mx.array(coords_np[surface_idx].astype(np.int32))


def _prune_outlier_gaussians(
    coords: mx.array,
    gs_dict: Dict[str, mx.array],
    density_radius: float = 2.0,
    min_neighbors: int = 3,
) -> Tuple[mx.array, Dict[str, mx.array], int]:
    """Remove isolated Gaussians via local voxel-density check.

    For each voxel, count neighbors within ``density_radius`` (in voxel-grid
    units, e.g. 2.0 means a Chebyshev/Euclidean ball of radius 2 voxels). If
    the neighborhood (including self) holds fewer than ``min_neighbors``
    points the voxel is treated as a floating outlier and ALL G Gaussians
    associated with that voxel are dropped from ``gs_dict``.

    The decoder emits each ``gs_dict`` entry with shape ``(N * G, ...)``
    laid out voxel-major (G consecutive Gaussians per voxel). We infer G
    from the first entry and apply the same boolean mask to every entry.

    Args:
        coords: (N, 4) int voxel coords [b, x, y, z] on a 64^3 grid.
        gs_dict: dict of Gaussian params, each with leading dim ``N * G``.
        density_radius: ball radius in voxel-grid units.
        min_neighbors: keep voxel iff neighbor count (including self) >= this.

    Returns:
        (filtered_coords, filtered_gs_dict, n_pruned)  -- ``n_pruned`` is the
        number of voxels removed (NOT Gaussians).
    """
    coords_np = np.asarray(coords).astype(np.int64)
    N = int(coords_np.shape[0])
    if N == 0 or not gs_dict:
        return coords, gs_dict, 0

    # Infer G from the first entry: leading dim is N * G.
    first_key = next(iter(gs_dict))
    first_val = gs_dict[first_key]
    M = int(first_val.shape[0])
    if M % N != 0:
        # Inconsistent shapes -- bail out without pruning rather than corrupt.
        print(
            f"[prune] WARN: gs '{first_key}' leading={M} not multiple of "
            f"N={N}; skipping outlier prune."
        )
        return coords, gs_dict, 0
    G = M // N

    # KDTree on (x, y, z); drop batch dim. Use scipy when available, else
    # fall back to numpy O(N^2) (still cheap at N <= ~16k).
    xyz = coords_np[:, 1:].astype(np.float32)
    keep_mask: np.ndarray
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz)
        # query_ball_point returns (for each query) the list of indices
        # within radius -- this list ALWAYS contains the query point itself,
        # so a count >= min_neighbors means at least (min_neighbors-1) other
        # voxels lie in the ball (matches the spec).
        nbrs = tree.query_ball_point(xyz, r=float(density_radius))
        counts = np.fromiter((len(n) for n in nbrs), dtype=np.int32, count=N)
        keep_mask = counts >= int(min_neighbors)
    except Exception:
        # Numpy fallback: pairwise squared distances, chunked to bound memory.
        r2 = float(density_radius) ** 2
        counts = np.zeros((N,), dtype=np.int32)
        chunk = 2048
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            d2 = np.sum((xyz[i0:i1, None, :] - xyz[None, :, :]) ** 2, axis=-1)
            counts[i0:i1] = np.sum(d2 <= r2, axis=1)
        keep_mask = counts >= int(min_neighbors)

    n_pruned = int((~keep_mask).sum())
    if n_pruned == 0:
        return coords, gs_dict, 0

    # Filter coords (voxel-level mask).
    coords_kept = coords_np[keep_mask]
    coords_out = mx.array(coords_kept.astype(np.int32))

    # Expand keep_mask to Gaussian-level ((N*G,) by repeating each voxel G times).
    gauss_mask = np.repeat(keep_mask, G)
    gauss_idx = mx.array(np.nonzero(gauss_mask)[0].astype(np.int32))

    gs_out: Dict[str, mx.array] = {}
    for k, v in gs_dict.items():
        # All entries share leading dim N*G with same voxel-major layout.
        gs_out[k] = v[gauss_idx]
    return coords_out, gs_out, n_pruned


def _avg_pool_coords(coords: mx.array, factor: int = 2) -> Tuple[mx.array, mx.array]:
    """Compute downsampled coords + per-input idx (numpy hot path; tiny N).

    Returns:
        coords_down: (N_down, 4) int32
        idx_up:      (N,) int32 mapping each input voxel to its bucket.
    """
    coords_np = np.asarray(coords).astype(np.int64)
    N = coords_np.shape[0]
    b = coords_np[:, 0]
    new_xyz = coords_np[:, 1:] // factor
    keys = (
        b.astype(np.int64) * (1 << 48)
        + (new_xyz[:, 0].astype(np.int64) << 32)
        + (new_xyz[:, 1].astype(np.int64) << 16)
        + new_xyz[:, 2].astype(np.int64)
    )
    uniq_keys, idx = np.unique(keys, return_inverse=True)
    new_b = (uniq_keys >> 48).astype(np.int32)
    new_x = ((uniq_keys >> 32) & 0xFFFF).astype(np.int32)
    new_y = ((uniq_keys >> 16) & 0xFFFF).astype(np.int32)
    new_z = (uniq_keys & 0xFFFF).astype(np.int32)
    new_coords = np.stack([new_b, new_x, new_y, new_z], axis=-1)
    return mx.array(new_coords), mx.array(idx.astype(np.int32))


def _scatter_mean(feats: mx.array, idx: mx.array, n_out: int) -> mx.array:
    """Average-pool feats into n_out groups using idx (per-input bucket assignment).

    feats: (N, C). idx: (N,). Returns (n_out, C).
    """
    feats_np = np.asarray(feats).astype(np.float32)
    idx_np = np.asarray(idx).astype(np.int64)
    C = feats_np.shape[1]
    out = np.zeros((n_out, C), dtype=np.float32)
    counts = np.zeros((n_out,), dtype=np.float32)
    np.add.at(out, idx_np, feats_np)
    np.add.at(counts, idx_np, 1.0)
    out = out / counts[:, None]
    return mx.array(out)


__all__ = [
    "SAM3DObjectsPipeline",
    "preprocess_image_to_mlx",
    "make_dummy_pointmap",
    "SLAT_STD",
    "SLAT_MEAN",
]
