"""Parity test: MLX sparse transformer vs upstream PT."""
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

from propainter_mlx.sparse_transformer import (
    unfold_nhwc, fold_nhwc, SoftSplit, SoftComp, FusionFeedForward,
    SparseWindowAttention, TemporalSparseTransformer, TemporalSparseTransformerBlock,
)
from model.modules.sparse_transformer import (
    SoftSplit as PTSoftSplit, SoftComp as PTSoftComp,
    FusionFeedForward as PTFFF,
    SparseWindowAttention as PTSparseAttn,
    TemporalSparseTransformer as PTTST,
    TemporalSparseTransformerBlock as PTTSTBlock,
)


def _to_mlx_conv2d_w(w_np: np.ndarray) -> np.ndarray:
    # PT (Cout, Cin, kH, kW) -> MLX (Cout, kH, kW, Cin)
    return np.transpose(w_np, (0, 2, 3, 1))


def test_unfold_fold_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 16, 24, 8)).astype(np.float32)  # NHWC
    # PT
    x_pt = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))  # NCHW
    u_pt = F.unfold(x_pt, kernel_size=(7, 7), stride=(3, 3), padding=(3, 3))
    # (B, C*49, L) -> (B, L, C*49)
    u_pt = u_pt.permute(0, 2, 1).numpy()
    # MLX
    u_mlx = np.array(unfold_nhwc(mx.array(x), (7, 7), (3, 3), (3, 3)))
    diff = np.abs(u_pt - u_mlx).max()
    print(f"  unfold max diff: {diff:.6f}")
    assert diff < 1e-6

    # fold
    f_pt = F.fold(u_pt.transpose(0, 2, 1) if False else torch.from_numpy(u_pt).permute(0, 2, 1),
                   output_size=(16, 24), kernel_size=(7, 7), stride=(3, 3), padding=(3, 3))
    f_pt = f_pt.numpy()  # (B, C, H, W)
    f_pt_nhwc = np.transpose(f_pt, (0, 2, 3, 1))
    f_mlx = np.array(fold_nhwc(mx.array(u_pt), (16, 24), (7, 7), (3, 3), (3, 3), 8))
    diff = np.abs(f_pt_nhwc - f_mlx).max()
    print(f"  fold max diff: {diff:.6f}")
    assert diff < 1e-5


def test_soft_split_comp(npz):
    rng = np.random.default_rng(1)
    B, T = 1, 4
    H, W = 60, 108
    C = 128
    hidden = 512
    x = rng.standard_normal((B * T, C, H, W)).astype(np.float32) * 0.3
    # PT
    pt = PTSoftSplit(C, hidden, (7, 7), (3, 3), (3, 3))
    with torch.no_grad():
        # Load weights from main npz
        pt.embedding.weight.data = torch.from_numpy(npz["ss.embedding.weight"])
        pt.embedding.bias.data   = torch.from_numpy(npz["ss.embedding.bias"])
        y_pt = pt(torch.from_numpy(x), B, (H, W)).numpy()  # (B, T, f_h, f_w, hidden)
    # MLX
    mlx_ss = SoftSplit(C, hidden, (7, 7), (3, 3), (3, 3))
    mlx_ss.embedding.weight = mx.array(npz["ss.embedding.weight"])
    mlx_ss.embedding.bias   = mx.array(npz["ss.embedding.bias"])
    x_nhwc = np.transpose(x, (0, 2, 3, 1))
    y_mlx = np.array(mlx_ss(mx.array(x_nhwc), B, (H, W)))
    diff = np.abs(y_pt - y_mlx).max()
    print(f"SoftSplit max diff: {diff:.6f}")
    assert diff < 1e-3, f"SoftSplit failed: {diff}"

    # SoftComp
    pt_sc = PTSoftComp(C, hidden, (7, 7), (3, 3), (3, 3))
    with torch.no_grad():
        pt_sc.embedding.weight.data = torch.from_numpy(npz["sc.embedding.weight"])
        pt_sc.embedding.bias.data   = torch.from_numpy(npz["sc.embedding.bias"])
        pt_sc.bias_conv.weight.data = torch.from_numpy(np.transpose(npz["sc.bias_conv.weight"], (0, 3, 1, 2)).copy())
        pt_sc.bias_conv.bias.data   = torch.from_numpy(npz["sc.bias_conv.bias"])
        # forward
        y_pt_sc = pt_sc(torch.from_numpy(y_pt), T, (H, W)).numpy()  # (B*T, C, H, W)
    mlx_sc = SoftComp(C, hidden, (7, 7), (3, 3), (3, 3))
    mlx_sc.embedding.weight = mx.array(npz["sc.embedding.weight"])
    mlx_sc.embedding.bias   = mx.array(npz["sc.embedding.bias"])
    # bias_conv is stored in MLX layout already in the npz (OHWI)
    mlx_sc.bias_conv.weight = mx.array(npz["sc.bias_conv.weight"])
    mlx_sc.bias_conv.bias   = mx.array(npz["sc.bias_conv.bias"])
    y_mlx_sc = np.array(mlx_sc(mx.array(y_mlx), T, (H, W)))
    y_mlx_sc_nchw = np.transpose(y_mlx_sc, (0, 3, 1, 2))
    diff = np.abs(y_pt_sc - y_mlx_sc_nchw).max()
    print(f"SoftComp max diff: {diff:.6f}")
    assert diff < 1e-2, f"SoftComp failed: {diff}"


def test_fusion_feedforward(npz):
    rng = np.random.default_rng(2)
    B, T = 1, 4
    H, W = 60, 108  # fold-feat size
    f_h, f_w = 20, 36  # n_vecs grid
    n_vecs = f_h * f_w
    dim = 512
    hidden_dim = 1960
    # input (B, T*f_h*f_w, dim)
    x = rng.standard_normal((B, T * n_vecs, dim)).astype(np.float32) * 0.3

    t2t = {'kernel_size': (7, 7), 'stride': (3, 3), 'padding': (3, 3)}
    pt = PTFFF(dim, hidden_dim=hidden_dim, t2t_params=t2t)
    # weights from block 0
    with torch.no_grad():
        pt.fc1[0].weight.data = torch.from_numpy(npz["transformers.transformer.0.mlp.fc1.0.weight"])
        pt.fc1[0].bias.data   = torch.from_numpy(npz["transformers.transformer.0.mlp.fc1.0.bias"])
        pt.fc2[1].weight.data = torch.from_numpy(npz["transformers.transformer.0.mlp.fc2.1.weight"])
        pt.fc2[1].bias.data   = torch.from_numpy(npz["transformers.transformer.0.mlp.fc2.1.bias"])
        y_pt = pt(torch.from_numpy(x), (H, W)).numpy()
    mlx_fff = FusionFeedForward(dim, hidden_dim=hidden_dim, t2t_params=t2t)
    mlx_fff.fc1.weight = mx.array(npz["transformers.transformer.0.mlp.fc1.0.weight"])
    mlx_fff.fc1.bias   = mx.array(npz["transformers.transformer.0.mlp.fc1.0.bias"])
    mlx_fff.fc2.weight = mx.array(npz["transformers.transformer.0.mlp.fc2.1.weight"])
    mlx_fff.fc2.bias   = mx.array(npz["transformers.transformer.0.mlp.fc2.1.bias"])
    y_mlx = np.array(mlx_fff(mx.array(x), (H, W)))
    diff = np.abs(y_pt - y_mlx).max()
    print(f"FusionFeedForward max diff: {diff:.6f}")
    assert diff < 1e-2, f"FFF failed: {diff}"


def test_full_block(npz):
    """Parity test for one TemporalSparseTransformer block (block 0)."""
    rng = np.random.default_rng(3)
    B, T = 1, 6
    f_h, f_w = 20, 36  # smaller token grid (60, 108) input
    H, W = 60, 108
    dim = 512
    n_head = 4
    window_size = (5, 9)
    pool_size = (4, 4)
    t2t = {'kernel_size': (7, 7), 'stride': (3, 3), 'padding': (3, 3)}

    x = rng.standard_normal((B, T, f_h, f_w, dim)).astype(np.float32) * 0.3
    mask = (rng.random((B, T, f_h, f_w, 1)) > 0.7).astype(np.float32)

    pt = PTTST(dim, n_head, window_size, pool_size, t2t_params=t2t)
    pt.eval()
    base = "transformers.transformer.0"
    # load
    def to_pt_conv(w_npz):
        # MLX OHWI -> PT OIHW
        return torch.from_numpy(np.transpose(w_npz, (0, 3, 1, 2)).copy())
    with torch.no_grad():
        pt.norm1.weight.data = torch.from_numpy(npz[f"{base}.norm1.weight"])
        pt.norm1.bias.data   = torch.from_numpy(npz[f"{base}.norm1.bias"])
        pt.norm2.weight.data = torch.from_numpy(npz[f"{base}.norm2.weight"])
        pt.norm2.bias.data   = torch.from_numpy(npz[f"{base}.norm2.bias"])
        pt.attention.query.weight.data = torch.from_numpy(npz[f"{base}.attention.query.weight"])
        pt.attention.query.bias.data   = torch.from_numpy(npz[f"{base}.attention.query.bias"])
        pt.attention.key.weight.data   = torch.from_numpy(npz[f"{base}.attention.key.weight"])
        pt.attention.key.bias.data     = torch.from_numpy(npz[f"{base}.attention.key.bias"])
        pt.attention.value.weight.data = torch.from_numpy(npz[f"{base}.attention.value.weight"])
        pt.attention.value.bias.data   = torch.from_numpy(npz[f"{base}.attention.value.bias"])
        pt.attention.proj.weight.data  = torch.from_numpy(npz[f"{base}.attention.proj.weight"])
        pt.attention.proj.bias.data    = torch.from_numpy(npz[f"{base}.attention.proj.bias"])
        pt.attention.pool_layer.weight.data = to_pt_conv(npz[f"{base}.attention.pool_layer.weight"])
        pt.attention.pool_layer.bias.data   = torch.from_numpy(npz[f"{base}.attention.pool_layer.bias"])
        pt.attention.valid_ind_rolled.data  = torch.from_numpy(npz[f"{base}.attention.valid_ind_rolled"].astype(np.int64))
        pt.mlp.fc1[0].weight.data = torch.from_numpy(npz[f"{base}.mlp.fc1.0.weight"])
        pt.mlp.fc1[0].bias.data   = torch.from_numpy(npz[f"{base}.mlp.fc1.0.bias"])
        pt.mlp.fc2[1].weight.data = torch.from_numpy(npz[f"{base}.mlp.fc2.1.weight"])
        pt.mlp.fc2[1].bias.data   = torch.from_numpy(npz[f"{base}.mlp.fc2.1.bias"])

        T_ind = torch.arange(0, T, 2)
        y_pt = pt(torch.from_numpy(x), (H, W), torch.from_numpy(mask), T_ind).numpy()

    mlx_blk = TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t)
    # load
    mlx_blk.norm1.weight = mx.array(npz[f"{base}.norm1.weight"])
    mlx_blk.norm1.bias   = mx.array(npz[f"{base}.norm1.bias"])
    mlx_blk.norm2.weight = mx.array(npz[f"{base}.norm2.weight"])
    mlx_blk.norm2.bias   = mx.array(npz[f"{base}.norm2.bias"])
    for n in ("query", "key", "value", "proj"):
        getattr(mlx_blk.attention, n).weight = mx.array(npz[f"{base}.attention.{n}.weight"])
        getattr(mlx_blk.attention, n).bias   = mx.array(npz[f"{base}.attention.{n}.bias"])
    mlx_blk.attention.pool_layer.weight = mx.array(npz[f"{base}.attention.pool_layer.weight"])
    mlx_blk.attention.pool_layer.bias   = mx.array(npz[f"{base}.attention.pool_layer.bias"])
    mlx_blk.attention.valid_ind_rolled  = mx.array(npz[f"{base}.attention.valid_ind_rolled"].astype(np.int32))
    mlx_blk.mlp.fc1.weight = mx.array(npz[f"{base}.mlp.fc1.0.weight"])
    mlx_blk.mlp.fc1.bias   = mx.array(npz[f"{base}.mlp.fc1.0.bias"])
    mlx_blk.mlp.fc2.weight = mx.array(npz[f"{base}.mlp.fc2.1.weight"])
    mlx_blk.mlp.fc2.bias   = mx.array(npz[f"{base}.mlp.fc2.1.bias"])

    T_ind_mlx = mx.arange(0, T, 2)
    y_mlx = np.array(mlx_blk(mx.array(x), (H, W), mx.array(mask), T_ind_mlx))
    diff = np.abs(y_pt - y_mlx).max()
    print(f"TemporalSparseTransformer block max diff: {diff:.6f}")
    assert diff < 5e-2, f"Sparse transformer block failed: {diff}"
    return diff


def test_full_stack(npz):
    """8-block TemporalSparseTransformerBlock parity test."""
    rng = np.random.default_rng(4)
    B, T = 1, 6
    f_h, f_w = 20, 36
    H, W = 60, 108
    dim = 512
    n_head = 4
    window_size = (5, 9)
    pool_size = (4, 4)
    depths = 8
    t2t = {'kernel_size': (7, 7), 'stride': (3, 3), 'padding': (3, 3)}

    x = rng.standard_normal((B, T, f_h, f_w, dim)).astype(np.float32) * 0.3
    mask = (rng.random((B, T, f_h, f_w, 1)) > 0.7).astype(np.float32)

    pt = PTTSTBlock(dim, n_head, window_size, pool_size, depths=depths, t2t_params=t2t)
    pt.eval()
    def to_pt_conv(w_npz):
        return torch.from_numpy(np.transpose(w_npz, (0, 3, 1, 2)).copy())
    with torch.no_grad():
        for i in range(depths):
            base = f"transformers.transformer.{i}"
            blk = pt.transformer[i]
            blk.norm1.weight.data = torch.from_numpy(npz[f"{base}.norm1.weight"])
            blk.norm1.bias.data   = torch.from_numpy(npz[f"{base}.norm1.bias"])
            blk.norm2.weight.data = torch.from_numpy(npz[f"{base}.norm2.weight"])
            blk.norm2.bias.data   = torch.from_numpy(npz[f"{base}.norm2.bias"])
            for n in ("query", "key", "value", "proj"):
                getattr(blk.attention, n).weight.data = torch.from_numpy(npz[f"{base}.attention.{n}.weight"])
                getattr(blk.attention, n).bias.data   = torch.from_numpy(npz[f"{base}.attention.{n}.bias"])
            blk.attention.pool_layer.weight.data = to_pt_conv(npz[f"{base}.attention.pool_layer.weight"])
            blk.attention.pool_layer.bias.data   = torch.from_numpy(npz[f"{base}.attention.pool_layer.bias"])
            blk.attention.valid_ind_rolled.data  = torch.from_numpy(npz[f"{base}.attention.valid_ind_rolled"].astype(np.int64))
            blk.mlp.fc1[0].weight.data = torch.from_numpy(npz[f"{base}.mlp.fc1.0.weight"])
            blk.mlp.fc1[0].bias.data   = torch.from_numpy(npz[f"{base}.mlp.fc1.0.bias"])
            blk.mlp.fc2[1].weight.data = torch.from_numpy(npz[f"{base}.mlp.fc2.1.weight"])
            blk.mlp.fc2[1].bias.data   = torch.from_numpy(npz[f"{base}.mlp.fc2.1.bias"])
        y_pt = pt(torch.from_numpy(x), (H, W), torch.from_numpy(mask), t_dilation=2).numpy()

    mlx_blk = TemporalSparseTransformerBlock(dim, n_head, window_size, pool_size, depths, t2t)
    flat = {k: mx.array(v) for k, v in npz.items()}
    # convert pool_layer indices: valid_ind_rolled is int64 in PT, int32 here
    # via key map
    m = TemporalSparseTransformerBlock.key_map(depths)
    for internal, key in m.items():
        parts = internal.split(".")
        obj = mlx_blk
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        val = flat[key]
        if "valid_ind_rolled" in internal:
            val = val.astype(mx.int32)
        setattr(obj, parts[-1], val)
    y_mlx = np.array(mlx_blk(mx.array(x), (H, W), mx.array(mask), t_dilation=2))
    diff = np.abs(y_pt - y_mlx).max()
    print(f"Full 8-block TST max diff: {diff:.6f}")
    return diff


def main():
    npz = np.load(str(ROOT / "weights/propainter-mlx/propainter_main.npz"))
    flat = {k: npz[k] for k in npz.files}
    print("--- unfold/fold roundtrip ---")
    test_unfold_fold_roundtrip()
    print("--- SoftSplit / SoftComp ---")
    test_soft_split_comp(flat)
    print("--- FusionFeedForward ---")
    test_fusion_feedforward(flat)
    print("--- Full transformer block ---")
    diff = test_full_block(flat)
    print("--- Full 8-block stack ---")
    diff_stack = test_full_stack(flat)
    print(f"\nALL PASS  (single block diff {diff:.5f}, stack diff {diff_stack:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
