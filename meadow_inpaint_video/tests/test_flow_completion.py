"""Numerical parity test: MLX RecurrentFlowCompletion vs upstream PT."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ProPainter"))

import torch

from propainter_mlx.flow_completion import RecurrentFlowCompleteNet as MLXRFC
from model.recurrent_flow_completion import RecurrentFlowCompleteNet as PTRFC


def main():
    pt_weights = str(ROOT / "weights/propainter-pt/recurrent_flow_completion.pth")
    mlx_weights = str(ROOT / "weights/propainter-mlx/rfc.npz")

    # build PT
    pt = PTRFC()
    sd = torch.load(pt_weights, map_location="cpu", weights_only=True)
    pt.load_state_dict(sd, strict=True)
    pt.eval()

    # build MLX
    mlx_rfc = MLXRFC()
    n_loaded = mlx_rfc.load_npz(mlx_weights)
    print(f"loaded {n_loaded} MLX RFC tensors")

    # random flow / mask input
    rng = np.random.default_rng(0)
    B, T, H, W = 1, 3, 64, 80
    flows = rng.standard_normal((B, T - 1, 2, H, W)).astype(np.float32) * 2
    masks = (rng.random((B, T - 1, 1, H, W)) > 0.6).astype(np.float32)

    with torch.no_grad():
        pred_pt, _ = pt(torch.from_numpy(flows), torch.from_numpy(masks))
    pred_pt = pred_pt.numpy()  # (B, T-1, 2, H, W)

    flows_m = mx.array(flows)
    masks_m = mx.array(masks)
    pred_mlx = np.array(mlx_rfc.forward(flows_m, masks_m))

    print(f"pred shape: pt={pred_pt.shape}  mlx={pred_mlx.shape}")
    print(f"PT  range: [{pred_pt.min():.4f}, {pred_pt.max():.4f}]  mean |.| {np.abs(pred_pt).mean():.4f}")
    print(f"MLX range: [{pred_mlx.min():.4f}, {pred_mlx.max():.4f}]  mean |.| {np.abs(pred_mlx).mean():.4f}")
    diff = np.abs(pred_pt - pred_mlx)
    print(f"max abs diff: {diff.max():.5f}  mean abs diff: {diff.mean():.5f}")

    # MAE in masked region
    m = np.broadcast_to(masks, (B, T - 1, 2, H, W))
    masked_mae = (diff * m).sum() / max(m.sum(), 1)
    print(f"MAE in masked region: {masked_mae:.5f}")

    gate = 1.0
    if masked_mae < gate:
        print(f"PASS (MAE < {gate})")
        return 0
    print(f"FAIL (MAE >= {gate})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
