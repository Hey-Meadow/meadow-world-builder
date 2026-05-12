"""Parity test: MLX BidirectionalPropagation (learnable=True, channel=128)
vs upstream PT."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ProPainter"))

import torch

from propainter_mlx.feat_prop import (
    BidirectionalPropagation as MLXBP, flow_warp as mlx_flow_warp,
    fb_consistency_check as mlx_fb,
)
from model.propainter import BidirectionalPropagation as PTBP
from model.modules.flow_loss_utils import flow_warp as pt_flow_warp


def test_flow_warp():
    rng = np.random.default_rng(0)
    B, H, W, C = 1, 24, 32, 5
    x = rng.standard_normal((B, H, W, C)).astype(np.float32)
    flow = (rng.standard_normal((B, H, W, 2)).astype(np.float32) * 4)
    # PT
    x_pt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))
    flow_pt = torch.from_numpy(flow)  # (B, H, W, 2)
    y_pt = pt_flow_warp(x_pt, flow_pt)
    y_pt = np.transpose(y_pt.numpy(), (0, 2, 3, 1))
    y_mlx = np.array(mlx_flow_warp(mx.array(x), mx.array(flow)))
    diff = np.abs(y_pt - y_mlx).max()
    print(f"flow_warp max diff: {diff:.6f}")
    assert diff < 1e-4


def test_feat_prop(npz):
    rng = np.random.default_rng(1)
    B, T = 1, 5
    H, W = 30, 54   # quarter-res for 120x216 input
    C = 128

    x = rng.standard_normal((B, T, H, W, C)).astype(np.float32) * 0.3
    flows_fwd = rng.standard_normal((B, T - 1, H, W, 2)).astype(np.float32) * 1.5
    flows_bwd = rng.standard_normal((B, T - 1, H, W, 2)).astype(np.float32) * 1.5
    # mask: ProPainter uses (B, T, H, W, 2) for prop_mask_in (concat of two masks)
    mask = (rng.random((B, T, H, W, 2)) > 0.7).astype(np.float32)

    pt = PTBP(C, learnable=True)
    pt.eval()
    sd = torch.load(str(ROOT / "weights/propainter-pt/ProPainter.pth"), map_location="cpu", weights_only=True)
    # filter feat_prop_module
    sub_sd = {k[len("feat_prop_module."):]: v for k, v in sd.items() if k.startswith("feat_prop_module.")}
    pt.load_state_dict(sub_sd, strict=True)

    # PT forward expects (B, T, C, H, W), flow (B, T-1, 2, H, W), mask (B, T, mc, H, W)
    x_pt = torch.from_numpy(np.transpose(x, (0, 1, 4, 2, 3)))
    ff_pt = torch.from_numpy(np.transpose(flows_fwd, (0, 1, 4, 2, 3)))
    fb_pt = torch.from_numpy(np.transpose(flows_bwd, (0, 1, 4, 2, 3)))
    m_pt = torch.from_numpy(np.transpose(mask, (0, 1, 4, 2, 3)))
    with torch.no_grad():
        ob_pt, of_pt, out_pt, _ = pt(x_pt, ff_pt, fb_pt, m_pt, interpolation="bilinear")
    ob_pt = np.transpose(ob_pt.numpy(), (0, 1, 3, 4, 2))
    of_pt = np.transpose(of_pt.numpy(), (0, 1, 3, 4, 2))
    out_pt = np.transpose(out_pt.numpy(), (0, 1, 3, 4, 2))

    mlx_bp = MLXBP(C, learnable=True)
    mlx_bp.load_from_flat({k: mx.array(npz[k]) for k in npz.files})

    ob_m, of_m, out_m, _ = mlx_bp(mx.array(x), mx.array(flows_fwd),
                                    mx.array(flows_bwd), mx.array(mask),
                                    interpolation="bilinear")
    diff_ob = np.abs(ob_pt - np.array(ob_m)).max()
    diff_of = np.abs(of_pt - np.array(of_m)).max()
    diff_out = np.abs(out_pt - np.array(out_m)).max()
    print(f"feat_prop  outputs_b max diff: {diff_ob:.6f}")
    print(f"feat_prop  outputs_f max diff: {diff_of:.6f}")
    print(f"feat_prop  outputs   max diff: {diff_out:.6f}")
    assert diff_out < 1e-2, f"feat_prop failed: {diff_out}"


def main():
    npz = np.load(str(ROOT / "weights/propainter-mlx/propainter_main.npz"))
    print("--- flow_warp parity ---")
    test_flow_warp()
    print("--- feat_prop_module (learnable=True, C=128) ---")
    test_feat_prop(npz)
    print("\nPASS")


if __name__ == "__main__":
    main()
