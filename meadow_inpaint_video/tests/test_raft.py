"""Numerical parity test: MLX RAFT vs upstream PT RAFT."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ProPainter"))

import torch
import torch.nn.functional as F
import argparse

from propainter_mlx.raft import RAFT as MLXRaft, InputPadder as MLXPadder

# Load upstream PT RAFT
from RAFT.raft import RAFT as PTRaft
from RAFT.utils.utils import InputPadder as PTPadder


def make_pt_raft(weights_pth: str) -> PTRaft:
    class A:
        small = False
        mixed_precision = False
        alternate_corr = False
        dropout = 0.0
        def _get_kwargs(self): return [('small', False), ('mixed_precision', False),
                                       ('alternate_corr', False), ('dropout', 0.0)]
    model = PTRaft(A())
    sd = torch.load(weights_pth, map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def random_frame_pair(H=240, W=320, seed=0):
    """Smooth textured image with a small known horizontal translation."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # base = smooth sinusoid + low-freq noise
    noise = rng.standard_normal((H // 16 + 1, W // 16 + 1)).astype(np.float32)
    # upsample noise by repeat
    noise = np.kron(noise, np.ones((16, 16), dtype=np.float32))[:H, :W]
    base = (
        80 + 60 * np.sin(2 * np.pi * xx / 64)
        + 40 * np.cos(2 * np.pi * yy / 48)
        + 30 * noise
    )
    base = np.clip(base, 0, 255).astype(np.float32)
    a = np.stack([base, base, base], axis=0)[None]  # (1, 3, H, W)
    # synthesize a small horizontal translation by 5 px
    b = np.zeros_like(a)
    b[..., :, 5:] = a[..., :, :-5]
    b[..., :, :5] = a[..., :, :1]  # replicate
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-weights", default=str(ROOT / "weights/propainter-pt/raft-things.pth"))
    ap.add_argument("--mlx-weights", default=str(ROOT / "weights/propainter-mlx/raft.npz"))
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=320)
    args = ap.parse_args()

    a_np, b_np = random_frame_pair(args.height, args.width, seed=0)

    # --- PT inference ---
    pt = make_pt_raft(args.pt_weights)
    a_pt = torch.from_numpy(a_np)
    b_pt = torch.from_numpy(b_np)
    a_pt_n = 2 * (a_pt / 255.0) - 1.0
    b_pt_n = 2 * (b_pt / 255.0) - 1.0
    padder = PTPadder(a_pt_n.shape)
    a_pt_p, b_pt_p = padder.pad(a_pt_n, b_pt_n)
    with torch.no_grad():
        _, flow_pt_up = pt(a_pt_p, b_pt_p, iters=args.iters, test_mode=True)
    flow_pt = padder.unpad(flow_pt_up).numpy()  # (1, 2, H, W)
    flow_pt_hwc = np.transpose(flow_pt, (0, 2, 3, 1))  # (1, H, W, 2)

    # --- MLX inference ---
    mlx_raft = MLXRaft()
    n_loaded = mlx_raft.load_npz(args.mlx_weights)
    print(f"loaded {n_loaded} MLX RAFT tensors")

    # NHWC, normalised, padded
    a_mlx = mx.array(np.transpose(a_np, (0, 2, 3, 1)))
    b_mlx = mx.array(np.transpose(b_np, (0, 2, 3, 1)))
    a_mlx = 2 * (a_mlx / 255.0) - 1.0
    b_mlx = 2 * (b_mlx / 255.0) - 1.0
    padder_mlx = MLXPadder(a_mlx.shape)
    a_mlx_p, b_mlx_p = padder_mlx.pad(a_mlx, b_mlx)
    _, flow_mlx_up = mlx_raft(a_mlx_p, b_mlx_p, iters=args.iters)
    flow_mlx_up = padder_mlx.unpad(flow_mlx_up)
    flow_mlx = np.array(flow_mlx_up)  # NHWC

    # --- compare ---
    epe = np.sqrt(((flow_pt_hwc - flow_mlx) ** 2).sum(axis=-1)).mean()
    max_abs = np.abs(flow_pt_hwc - flow_mlx).max()
    print(f"flow shape: pt={flow_pt_hwc.shape} mlx={flow_mlx.shape}")
    print(f"PT  flow range: [{flow_pt_hwc.min():.3f}, {flow_pt_hwc.max():.3f}]  mean |.| = {np.abs(flow_pt_hwc).mean():.3f}")
    print(f"MLX flow range: [{flow_mlx.min():.3f}, {flow_mlx.max():.3f}]  mean |.| = {np.abs(flow_mlx).mean():.3f}")
    print(f"end-point error: {epe:.4f} px   max abs diff: {max_abs:.4f}")
    gate = 0.1
    if epe < gate:
        print(f"PASS (EPE < {gate})")
        return 0
    else:
        print(f"FAIL (EPE >= {gate})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
