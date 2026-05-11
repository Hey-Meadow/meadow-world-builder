"""Smoke + numerical tests for the MoGe MLX port.

Run with:
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        -m pytest meadow3d/tests/test_moge.py -v -s
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(scope="module")
def model():
    from meadow3d.models.moge_mlx import MoGeModel
    return MoGeModel.from_pretrained()


def test_smoke_forward(model):
    """Random input -> sensible shapes + no NaNs."""
    rng = np.random.default_rng(0)
    img = rng.random((1, 518, 518, 3), dtype=np.float32)
    out = model(mx.array(img))
    mx.eval(*[v for v in out.values() if isinstance(v, mx.array)])

    assert out["points"].shape == (1, 3, 518, 518)
    assert out["mask"].shape == (1, 1, 518, 518)
    assert out["intrinsics"].shape == (1, 3, 3)
    assert out["depth"].shape == (1, 518, 518)

    intr = np.asarray(out["intrinsics"])[0]
    assert intr[0, 0] > 0  # fx
    assert intr[1, 1] > 0  # fy
    assert intr[0, 2] == 0.5
    assert intr[1, 2] == 0.5
    assert intr[2, 2] == 1.0


def test_real_image_kidsroom(model):
    """Real kidsroom image -> sensible depth and mask coverage."""
    from PIL import Image

    img_path = os.path.join(
        _REPO_ROOT,
        "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png",
    )
    mask_path = os.path.join(
        _REPO_ROOT,
        "notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png",
    )
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask_bin = (mask > 0).astype(np.uint8)

    # Crop around mask + square pad + resize 518.
    ys, xs = np.nonzero(mask_bin)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    crop = rgb[y1:y2, x1:x2] * mask_bin[y1:y2, x1:x2, None]
    H, W = crop.shape[:2]
    S = max(H, W)
    crop = np.pad(
        crop,
        ((((S - H) // 2), (S - H - (S - H) // 2)),
         (((S - W) // 2), (S - W - (S - W) // 2)),
         (0, 0)),
        constant_values=0,
    )
    img = np.asarray(Image.fromarray(crop).resize((518, 518), Image.BICUBIC))
    img = img.astype(np.float32) / 255.0

    t0 = time.time()
    out = model(mx.array(img[None]))
    mx.eval(*[v for v in out.values() if isinstance(v, mx.array)])
    dt = time.time() - t0
    print(f"\n  forward time: {dt:.2f} s")

    pts = np.asarray(out["points"])[0]
    valid = np.isfinite(pts).all(axis=0)
    n_valid = valid.sum()
    print(f"  valid pixels: {n_valid} / {518*518}")
    # Expected ~170k valid pixels for the kidsroom mask (verified vs PT).
    assert n_valid > 100_000
    # z should be in a sensible camera-space range (1m..10m typical).
    z_valid = pts[2][valid]
    print(f"  z range: [{z_valid.min():.3f}, {z_valid.max():.3f}], "
          f"mean={z_valid.mean():.3f}")
    assert 0.5 < z_valid.mean() < 20.0


def test_match_pt_within_tolerance(model):
    """Numerical sanity vs PT MoGe on the kidsroom image."""
    from moge.model.v1 import MoGeModel as PTModel
    import torch
    from PIL import Image

    pt_model = PTModel.from_pretrained("Ruicheng/moge-vitl")
    pt_model.eval()

    img_path = os.path.join(
        _REPO_ROOT,
        "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png",
    )
    mask_path = os.path.join(
        _REPO_ROOT,
        "notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png",
    )
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., -1]
    mask_bin = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask_bin)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    crop = rgb[y1:y2, x1:x2] * mask_bin[y1:y2, x1:x2, None]
    H, W = crop.shape[:2]
    S = max(H, W)
    crop = np.pad(crop,
                  ((((S - H) // 2), (S - H - (S - H) // 2)),
                   (((S - W) // 2), (S - W - (S - W) // 2)),
                   (0, 0)), constant_values=0)
    img = np.asarray(Image.fromarray(crop).resize((518, 518), Image.BICUBIC))
    img = img.astype(np.float32) / 255.0

    mlx_out = model(mx.array(img[None]))
    mx.eval(*[v for v in mlx_out.values() if isinstance(v, mx.array)])
    pt_in = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        pt_out = pt_model.infer(pt_in, force_projection=False,
                                apply_mask=True, use_fp16=False)

    mlx_pts = np.asarray(mlx_out["points"])[0]
    pt_pts = pt_out["points"].numpy()
    if pt_pts.ndim == 3:
        pt_pts = np.transpose(pt_pts, (2, 0, 1))
    else:
        pt_pts = np.transpose(pt_pts[0], (2, 0, 1))

    mlx_valid = np.isfinite(mlx_pts).all(axis=0)
    pt_valid = np.isfinite(pt_pts).all(axis=0)
    common = mlx_valid & pt_valid
    diff_z = np.abs(mlx_pts[2][common] - pt_pts[2][common])
    print(f"\n  mlx_valid: {mlx_valid.sum()}, pt_valid: {pt_valid.sum()}")
    print(f"  z |diff| mean {diff_z.mean():.3f}, max {diff_z.max():.3f}")
    # Tolerance accounts for bicubic-antialias vs bilinear cubic resize differences
    # accumulated through 24 transformer blocks + 3 upsample stages.
    assert diff_z.mean() < 0.2
    assert diff_z.max() < 1.0
