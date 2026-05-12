"""Numerical parity test: MLX Encoder + Decoder vs PT InpaintGenerator parts."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ProPainter"))

import torch

from propainter_mlx.encoder import Encoder as MLXEnc
from propainter_mlx.decoder import Decoder as MLXDec
from model.propainter import Encoder as PTEnc, InpaintGenerator as PTInpaint


def main():
    pt_weights = str(ROOT / "weights/propainter-pt/ProPainter.pth")
    sd = torch.load(pt_weights, map_location="cpu", weights_only=True)

    # build PT encoder + decoder
    pt_ig = PTInpaint(init_weights=False)
    pt_ig.load_state_dict(sd, strict=True)
    pt_ig.eval()

    rng = np.random.default_rng(0)
    bt, H, W = 2, 256, 320  # H, W divisible by 4
    x = rng.standard_normal((bt, 5, H, W)).astype(np.float32) * 0.5
    with torch.no_grad():
        enc_pt = pt_ig.encoder(torch.from_numpy(x)).numpy()  # (bt, 128, H/4, W/4)

    # MLX
    mlx_enc = MLXEnc()
    npz = np.load(str(ROOT / "weights/propainter-mlx/propainter_main.npz"))
    flat = {k: mx.array(npz[k]) for k in npz.files}
    mlx_enc.load_from_flat(flat)
    x_m = mx.array(np.transpose(x, (0, 2, 3, 1)))
    enc_mlx = np.array(mlx_enc(x_m))
    enc_mlx_n = np.transpose(enc_mlx, (0, 3, 1, 2))
    diff = np.abs(enc_pt - enc_mlx_n)
    print(f"Encoder: pt shape {enc_pt.shape}  mlx {enc_mlx_n.shape}")
    print(f"  diff max {diff.max():.5f}  mean {diff.mean():.5f}")
    enc_pass = diff.max() < 1e-2

    # Decoder
    dec_in = rng.standard_normal((bt, 128, H // 4, W // 4)).astype(np.float32)
    with torch.no_grad():
        dec_pt = pt_ig.decoder(torch.from_numpy(dec_in)).numpy()
    mlx_dec = MLXDec()
    mlx_dec.load_from_flat(flat)
    dec_in_m = mx.array(np.transpose(dec_in, (0, 2, 3, 1)))
    dec_mlx = np.array(mlx_dec(dec_in_m))
    dec_mlx_n = np.transpose(dec_mlx, (0, 3, 1, 2))
    diff = np.abs(dec_pt - dec_mlx_n)
    print(f"Decoder: pt shape {dec_pt.shape}  mlx {dec_mlx_n.shape}")
    print(f"  diff max {diff.max():.5f}  mean {diff.mean():.5f}")
    dec_pass = diff.max() < 1e-2

    return 0 if (enc_pass and dec_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
