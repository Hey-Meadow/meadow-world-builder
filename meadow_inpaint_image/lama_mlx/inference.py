"""Single-image inpainting helper. Pads input to multiple of 8 (matches PT default)."""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .generator import FFCResNetGenerator


def _pad_to_multiple(img: np.ndarray, modulo: int = 8):
    """img: (H, W, C) float32. Pad bottom/right to multiple of modulo. Returns (img_padded, (H, W))."""
    H, W = img.shape[:2]
    Hp = ((H + modulo - 1) // modulo) * modulo
    Wp = ((W + modulo - 1) // modulo) * modulo
    if Hp == H and Wp == W:
        return img, (H, W)
    pad = np.zeros((Hp, Wp) + img.shape[2:], dtype=img.dtype)
    pad[:H, :W] = img
    return pad, (H, W)


def inpaint(model: FFCResNetGenerator, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint a single image.

    Args:
      model: loaded FFCResNetGenerator (eval mode)
      image_rgb: (H, W, 3) uint8 or float in [0,1]
      mask:      (H, W)   uint8 (0 = keep, >0 = remove)
    Returns:
      (H, W, 3) uint8 inpainted image
    """
    if image_rgb.dtype == np.uint8:
        img = image_rgb.astype(np.float32) / 255.0
    else:
        img = image_rgb.astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = (mask > 127).astype(np.float32) if mask.dtype == np.uint8 else (mask > 0.5).astype(np.float32)

    H, W = img.shape[:2]
    img_p, (h0, w0) = _pad_to_multiple(img, 8)
    mask_p, _ = _pad_to_multiple(m[..., None], 8)
    mask_p = mask_p[..., 0]

    # Build (1, 4, Hp, Wp) NCHW input: RGB * (1 - mask) for the visible region, then concat mask.
    # The PT inference pipeline (bin/predict.py) does this same masking.
    rgb = img_p * (1.0 - mask_p[..., None])
    x = np.concatenate([rgb.transpose(2, 0, 1), mask_p[None, ...]], axis=0)[None, ...].astype(np.float32)

    y = model(mx.array(x))
    mx.eval(y)
    y_np = np.array(y)[0].transpose(1, 2, 0)  # HWC
    y_np = np.clip(y_np, 0.0, 1.0)
    # Composite: keep original where mask==0, use prediction where mask==1
    final = img_p * (1.0 - mask_p[..., None]) + y_np * mask_p[..., None]
    final = final[:h0, :w0]
    return (final * 255.0 + 0.5).astype(np.uint8)
