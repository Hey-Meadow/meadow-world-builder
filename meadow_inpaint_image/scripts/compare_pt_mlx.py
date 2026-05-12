"""Run end-to-end inpainting with both PT and MLX, compute PSNR."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import mlx.core as mx

from _pt_helper import load_pt_generator
from lama_mlx.generator import FFCResNetGenerator
from lama_mlx.inference import inpaint


def pt_inpaint(net, img_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Mirror lama_mlx.inference.inpaint exactly, but with PT."""
    img = img_u8.astype(np.float32) / 255.0
    m = (mask_u8 > 127).astype(np.float32)
    H, W = img.shape[:2]
    # pad to mult 8
    Hp = ((H + 7) // 8) * 8
    Wp = ((W + 7) // 8) * 8
    img_p = np.zeros((Hp, Wp, 3), dtype=np.float32); img_p[:H, :W] = img
    mask_p = np.zeros((Hp, Wp), dtype=np.float32); mask_p[:H, :W] = m
    rgb = img_p * (1.0 - mask_p[..., None])
    x = np.concatenate([rgb.transpose(2, 0, 1), mask_p[None, ...]], axis=0)[None, ...].astype(np.float32)
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()[0].transpose(1, 2, 0)
    y = np.clip(y, 0, 1)
    final = img_p * (1.0 - mask_p[..., None]) + y * mask_p[..., None]
    final = final[:H, :W]
    return (final * 255 + 0.5).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def main():
    img = np.array(Image.open(ROOT / "tests" / "data" / "image.png").convert("RGB"))
    msk = np.array(Image.open(ROOT / "tests" / "data" / "mask.png").convert("L"))

    print("loading PT generator...")
    pt_net = load_pt_generator(ROOT / "weights" / "big-lama" / "models" / "best.ckpt")
    print("loading MLX generator...")
    mlx_net = FFCResNetGenerator.from_npz(str(ROOT / "weights" / "lama_mlx.npz"))

    t0 = time.time()
    pt_out = pt_inpaint(pt_net, img, msk)
    pt_dt = time.time() - t0
    print(f"PT inpaint:  {pt_dt:.2f}s")

    # warmup MLX
    _ = inpaint(mlx_net, img, msk)
    t0 = time.time()
    mlx_out = inpaint(mlx_net, img, msk)
    mlx_dt = time.time() - t0
    print(f"MLX inpaint: {mlx_dt:.2f}s")

    Image.fromarray(pt_out).save(ROOT / "tests" / "data" / "PT_out.png")
    Image.fromarray(mlx_out).save(ROOT / "tests" / "data" / "MLX_out.png")
    diff_img = np.abs(pt_out.astype(np.int16) - mlx_out.astype(np.int16)).astype(np.uint8)
    Image.fromarray(diff_img).save(ROOT / "tests" / "data" / "diff.png")
    val = psnr(pt_out, mlx_out)
    maxd = int(diff_img.max())
    print(f"PSNR (MLX vs PT): {val:.2f} dB  max-pixel-diff: {maxd}")

    # Also test on a 1024x1024 image for M1 timing
    print("\n--- timing at 1024x1024 ---")
    big = np.array(Image.fromarray(img).resize((1024, 1024)))
    big_msk = np.array(Image.fromarray(msk).resize((1024, 1024), Image.NEAREST))
    _ = inpaint(mlx_net, big, big_msk)  # warmup
    t0 = time.time()
    _ = inpaint(mlx_net, big, big_msk)
    print(f"MLX 1024x1024 inpaint: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
