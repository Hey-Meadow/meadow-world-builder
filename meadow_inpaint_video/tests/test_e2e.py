"""End-to-end parity test: MLX InpaintGenerator vs PT InpaintGenerator.

Runs both on synthetic 8-frame 64x96 input + mask and compares max-abs-diff.
"""
from __future__ import annotations
import sys
from pathlib import Path
import time
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ProPainter"))

import torch

from propainter_mlx.propainter import InpaintGenerator as MLXIG
from model.propainter import InpaintGenerator as PTIG


def main():
    pt_path = str(ROOT / "weights/propainter-pt/ProPainter.pth")
    mlx_path = str(ROOT / "weights/propainter-mlx/propainter_main.npz")

    pt = PTIG(init_weights=False)
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    pt.load_state_dict(sd, strict=True)
    pt.eval()

    mlx_ig = MLXIG.from_npz(mlx_path)

    rng = np.random.default_rng(0)
    B = 1
    l_t = 6           # number of local frames
    n_ref = 3
    T = l_t + n_ref
    H, W = 80, 144     # divisible by 4 and 8
    # generate masked frames (already pre-multiplied)
    frames = rng.standard_normal((B, T, 3, H, W)).astype(np.float32) * 0.3
    flows_f = rng.standard_normal((B, l_t - 1, 2, H, W)).astype(np.float32)
    flows_b = rng.standard_normal((B, l_t - 1, 2, H, W)).astype(np.float32)
    masks_in = (rng.random((B, T, 1, H, W)) > 0.7).astype(np.float32)
    masks_updated = (rng.random((B, T, 1, H, W)) > 0.7).astype(np.float32)

    with torch.no_grad():
        y_pt = pt(
            torch.from_numpy(frames),
            (torch.from_numpy(flows_f), torch.from_numpy(flows_b)),
            torch.from_numpy(masks_in),
            torch.from_numpy(masks_updated),
            l_t,
        ).numpy()

    t0 = time.time()
    y_mlx = mlx_ig(
        mx.array(frames),
        (mx.array(flows_f), mx.array(flows_b)),
        mx.array(masks_in),
        mx.array(masks_updated),
        l_t,
    )
    mx.eval(y_mlx)
    t_e = time.time() - t0
    y_mlx_np = np.array(y_mlx)

    print(f"PT  output: {y_pt.shape}")
    print(f"MLX output: {y_mlx_np.shape}")
    diff = np.abs(y_pt - y_mlx_np)
    print(f"E2E max abs diff: {diff.max():.5f}")
    print(f"E2E mean abs diff: {diff.mean():.5f}")
    print(f"E2E elapsed (MLX): {t_e:.2f}s   for B={B} T={T} H={H} W={W}, l_t={l_t}")
    gate = 5e-2
    if diff.max() < gate:
        print(f"PASS (max diff < {gate})")
        return 0
    print(f"FAIL (max diff >= {gate})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
