"""Auto-mask via MLX SAM3.

Wraps the community ``mlx_sam3`` package so callers can convert raw RGB
images into RGBA (with alpha = SAM3 mask) without touching SAM3 internals.

Usage:
    from meadow3d.utils.auto_mask import auto_mask_image

    rgba = auto_mask_image("photo.jpg", text_prompt="plush toy")
    # rgba: (H, W, 4) uint8 — pass directly to SAM3DObjectsPipeline

The first call downloads model weights (~3 GB, ~50 s on first run).
Subsequent calls reuse the cached model in process; per-image inference
~1.5 s on M1 Max.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


# Lazy-init singletons.
_MODEL = None
_PROCESSOR = None


def _ensure_model():
    """Load SAM3 model + processor once per process."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        _MODEL = build_sam3_image_model()
        _PROCESSOR = Sam3Processor(_MODEL, confidence_threshold=0.5)
    return _MODEL, _PROCESSOR


def auto_mask_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    text_prompt: str = "main object",
) -> np.ndarray:
    """Run SAM3 to generate a mask for the largest matching object.

    Args:
        image: Path to RGB/RGBA image, PIL Image, or HxWx3 / HxWx4 uint8 ndarray.
        text_prompt: Text guiding which object to segment. Default
            ``"main object"`` works for most single-subject photos.

    Returns:
        rgba: (H, W, 4) uint8 ndarray. RGB from input; alpha is the binary mask
        (255 inside, 0 outside) of the largest detected object.

    Raises:
        RuntimeError: if SAM3 finds zero matching objects.
    """
    # Normalise input to PIL RGB.
    if isinstance(image, (str, Path)):
        pil = Image.open(image)
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image
    rgb_pil = pil.convert("RGB")

    _, proc = _ensure_model()
    state = proc.set_image(rgb_pil)
    state = proc.set_text_prompt(text_prompt, state)

    masks = state.get("masks", [])
    scores = state.get("scores", [])
    if len(masks) == 0:
        raise RuntimeError(
            f"SAM3 found no objects matching prompt {text_prompt!r}. "
            f"Try a different prompt (e.g. the object's name)."
        )

    # Pick the largest matching mask. Each mask is MLX array of shape (1, H, W).
    masks_np = [np.array(m).squeeze() for m in masks]
    areas = [int(m.sum()) for m in masks_np]
    idx = int(np.argmax(areas))
    mask = masks_np[idx].astype(np.uint8) * 255

    rgb = np.array(rgb_pil, dtype=np.uint8)
    if mask.shape[:2] != rgb.shape[:2]:
        # Resize mask to match if SAM3 returned a different resolution.
        mh, mw = mask.shape[:2]
        rh, rw = rgb.shape[:2]
        if (mh, mw) != (rh, rw):
            mask = np.array(
                Image.fromarray(mask).resize((rw, rh), Image.NEAREST),
                dtype=np.uint8,
            )

    rgba = np.concatenate([rgb, mask[..., None]], axis=-1)
    return rgba
