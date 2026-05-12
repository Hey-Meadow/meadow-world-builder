"""Numerical-parity test for one FFC_BN_ACT block (MLX vs PyTorch).

Loads weights from a fresh PT module, copies them into the MLX module, and
asserts max abs diff < 1e-3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _build_modules(in_ch=64, out_ch=128, ratio_gin=0.5, ratio_gout=0.75, kernel=3, padding=1):
    import torch
    import mlx.core as mx
    from lama_mlx.ffc import FFC_BN_ACT as MLXFFCBA
    from _pt_helper import install_pl_stubs
    install_pl_stubs()
    if str(Path("/Users/akaihuangm1/Desktop/github/lama")) not in sys.path:
        sys.path.insert(0, "/Users/akaihuangm1/Desktop/github/lama")
    from saicinpainting.training.modules.ffc import FFC_BN_ACT as PTFFCBA
    import torch.nn as nn

    torch.manual_seed(42)
    pt = PTFFCBA(in_ch, out_ch, kernel_size=kernel, ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                 padding=padding, padding_type="reflect", enable_lfu=False, activation_layer=nn.ReLU)
    pt.eval()
    # Randomize BN running stats so they aren't trivial.
    for m in pt.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.data.uniform_(-0.5, 0.5)
            m.running_var.data.uniform_(0.5, 1.5)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.3, 0.3)
    mlx_mod = MLXFFCBA(in_ch, out_ch, kernel_size=kernel, ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                       padding=padding, activation=True, enable_lfu=False)
    return pt, mlx_mod


def _copy_weights(pt, mlx_mod):
    """Copy PT module state_dict into MLX module via parameter tree."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    # We need to map: PT key -> MLX key, plus transpose conv weights.
    pt_sd = pt.state_dict()

    def _conv(arr):
        return np.transpose(arr.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)

    new = {}
    for pk, v in pt_sd.items():
        if pk.endswith("num_batches_tracked"):
            continue
        # Parse
        # Possible PT keys:
        #   ffc.convl2l.weight, ffc.convl2g.weight, ffc.convg2l.weight
        #   ffc.convg2g.conv1.0.weight, ffc.convg2g.conv1.1.weight/bias/running_*
        #   ffc.convg2g.fu.conv_layer.weight, ffc.convg2g.fu.bn.{w,b,run}
        #   ffc.convg2g.conv2.weight
        #   bn_l.{w,b,run_mean,run_var}, bn_g.{...}
        parts = pk.split(".")
        leaf = parts[-1]
        if parts[0] == "ffc":
            if parts[1] in ("convl2l", "convl2g", "convg2l"):
                mlx_key = f"ffc.{parts[1]}.conv.weight"
                new[mlx_key] = mx.array(_conv(v))
                continue
            elif parts[1] == "convg2g":
                if parts[2] == "conv1":
                    if parts[3] == "0":
                        new["ffc.convg2g.conv1_conv.weight"] = mx.array(_conv(v))
                    elif parts[3] == "1":
                        # BN
                        new[f"ffc.convg2g.conv1_bn.{leaf}"] = mx.array(v.detach().cpu().numpy().astype(np.float32))
                    continue
                elif parts[2] == "fu":
                    if parts[3] == "conv_layer":
                        new["ffc.convg2g.fu.conv_layer.weight"] = mx.array(_conv(v))
                    elif parts[3] == "bn":
                        new[f"ffc.convg2g.fu.bn.{leaf}"] = mx.array(v.detach().cpu().numpy().astype(np.float32))
                    continue
                elif parts[2] == "conv2":
                    new["ffc.convg2g.conv2.weight"] = mx.array(_conv(v))
                    continue
        elif parts[0] in ("bn_l", "bn_g"):
            new[f"{parts[0]}.{leaf}"] = mx.array(v.detach().cpu().numpy().astype(np.float32))
            continue
    # Filter to only keys that exist in MLX param tree
    param_flat = dict(tree_flatten(mlx_mod.parameters()))
    new_filtered = {k: v for k, v in new.items() if k in param_flat}
    missing = [k for k in param_flat if k not in new_filtered]
    if missing:
        print("MISSING:", missing)
    mlx_mod.update(tree_unflatten(list(new_filtered.items())))
    mlx_mod.eval()
    return mlx_mod


def _run_pair(in_ch=64, out_ch=128, ratio_gin=0.5, ratio_gout=0.75, H=32, W=32, seed=0):
    import torch
    import mlx.core as mx
    pt, mlx_mod = _build_modules(in_ch, out_ch, ratio_gin, ratio_gout)
    _copy_weights(pt, mlx_mod)

    torch.manual_seed(seed)
    x = torch.randn(1, in_ch, H, W)
    in_cg = int(in_ch * ratio_gin)
    in_cl = in_ch - in_cg
    if in_cg == 0:
        x_l_pt = x
        x_g_pt = 0
    else:
        x_l_pt = x[:, :in_cl]
        x_g_pt = x[:, in_cl:]

    with torch.no_grad():
        y_l, y_g = pt((x_l_pt, x_g_pt))
    # MLX
    x_np = x.numpy()
    x_nhwc = np.transpose(x_np, (0, 2, 3, 1))
    x_l_m = mx.array(np.ascontiguousarray(x_nhwc[..., :in_cl])) if in_cl > 0 else None
    x_g_m = mx.array(np.ascontiguousarray(x_nhwc[..., in_cl:])) if in_cg > 0 else None
    y_l_m, y_g_m = mlx_mod(x_l_m, x_g_m)
    mx.eval(y_l_m if y_l_m is not None else mx.array(0.0))
    if y_g_m is not None:
        mx.eval(y_g_m)

    diffs = {}
    if isinstance(y_l, torch.Tensor):
        y_l_np = np.transpose(np.array(y_l_m), (0, 3, 1, 2))
        diffs["l"] = float(np.abs(y_l.numpy() - y_l_np).max())
    if isinstance(y_g, torch.Tensor):
        y_g_np = np.transpose(np.array(y_g_m), (0, 3, 1, 2))
        diffs["g"] = float(np.abs(y_g.numpy() - y_g_np).max())
    return diffs


def test_ffc_full_branches():
    """All 4 branches active: ratio_gin>0 and ratio_gout>0."""
    diffs = _run_pair(in_ch=64, out_ch=128, ratio_gin=0.5, ratio_gout=0.75)
    print("diffs:", diffs)
    for k, v in diffs.items():
        assert v < 1e-3, f"branch {k} diff {v} > 1e-3"


def test_ffc_local_only():
    """ratio_gin=0, ratio_gout=0: should be just a conv2d+bn+relu."""
    diffs = _run_pair(in_ch=4, out_ch=64, ratio_gin=0.0, ratio_gout=0.0)
    print("diffs:", diffs)
    assert diffs.get("l", 0) < 1e-3


if __name__ == "__main__":
    test_ffc_full_branches()
    test_ffc_local_only()
    print("OK")
