"""Tests for ``meadow_sb/utils`` shared primitives (Agent I).

Quality gate for the parallel sprint: every downstream agent (A/B/C) depends on
:class:`MLXBlock` matching a PyTorch reference within ``1e-5`` max-abs-diff for
fp32 primitive ops.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import mlx.core as mx
import torch
import torch.nn as nn

from meadow_sb.utils import (
    MLXBlock,
    assert_close,
    attach_weights,
    load_npz_module,
    mlx_to_np,
    pt_to_mlx,
)


# ---------------------------------------------------------------------------
# PyTorch reference Block — mirrors MLXBlock exactly for numerical comparison.
# Equivalent to timm `Block` + DINOv2 LayerScale; we inline it so the test
# does not depend on a specific timm version.
# ---------------------------------------------------------------------------


class _PTLayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x):
        return x * self.gamma


class _PTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _PTMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _PTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _PTAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.ls1 = _PTLayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _PTMlp(dim, int(dim * mlp_ratio))
        self.ls2 = _PTLayerScale(dim, init_values=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pt_to_mlx_round_trip():
    """torch.Tensor -> mx.array -> np.ndarray preserves values bit-exactly."""
    torch.manual_seed(0)
    t = torch.randn(3, 5, 7, dtype=torch.float32)
    m = pt_to_mlx(t)
    n = mlx_to_np(m)

    diff = float(np.abs(t.numpy() - n).max())
    print(f"[round_trip] max_abs_diff={diff:.3e}")
    assert diff == 0.0, f"round-trip not bit-exact: {diff}"
    assert n.shape == (3, 5, 7)
    assert n.dtype == np.float32


def test_pt_to_mlx_accepts_numpy_and_scalars():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert pt_to_mlx(arr).shape == (3, 4)
    assert pt_to_mlx(arr).dtype == mx.float32

    s = pt_to_mlx(1.5)
    assert s.shape == ()
    assert s.dtype == mx.float32

    # int -> int32 after downcast
    i = pt_to_mlx(np.array([1, 2, 3], dtype=np.int64))
    assert i.dtype == mx.int32


def test_block_matches_pytorch_reference():
    """Build identical MLX + PT blocks, transfer weights, compare forward pass."""
    torch.manual_seed(42)
    dim, num_heads, N, B = 64, 4, 17, 2

    pt_block = _PTBlock(dim, num_heads, mlp_ratio=4.0, init_values=1e-5).eval()

    # Materialize a state_dict and save to npz.
    sd = {k: v.detach().cpu().numpy() for k, v in pt_block.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "block.npz")
        np.savez(path, **sd)

        # Build the MLX block.
        mlx_block = MLXBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4.0,
            qkv_bias=True, proj_bias=True, qk_norm=False,
            init_values=1e-5, norm_eps=1e-6, act="gelu",
            use_cross_attn=False,
        )

        # Load npz, attach (keys already match: norm1.weight, attn.qkv.weight,
        # ls1.gamma, mlp.fc1.weight, ls2.gamma, etc.).
        weights = load_npz_module(path)
        diag = attach_weights(mlx_block, weights, strict=True)
        assert diag["missing"] == []
        assert diag["unexpected"] == []

    # Forward pass on identical input
    x_np = np.random.RandomState(0).randn(B, N, dim).astype(np.float32)
    pt_out = pt_block(torch.from_numpy(x_np))
    mlx_out = mlx_block(pt_to_mlx(x_np))

    max_abs = assert_close(mlx_out, pt_out, atol=1e-5, name="MLXBlock-vs-PT")
    print(f"[block_match] max_abs_diff={max_abs:.3e}")


def test_block_with_cross_attn_runs():
    """Smoke-test: cross-attention variant builds, loads, and forwards."""
    dim, ctx_dim, num_heads, N, M, B = 64, 80, 4, 9, 13, 2
    mlx_block = MLXBlock(
        dim=dim, num_heads=num_heads, mlp_ratio=4.0,
        init_values=1e-5, use_cross_attn=True, ctx_dim=ctx_dim,
    )
    x = pt_to_mlx(np.random.RandomState(1).randn(B, N, dim).astype(np.float32))
    ctx = pt_to_mlx(np.random.RandomState(2).randn(B, M, ctx_dim).astype(np.float32))
    y = mlx_block(x, context=ctx)
    assert y.shape == (B, N, dim)
    mx.eval(y)


if __name__ == "__main__":
    test_pt_to_mlx_round_trip()
    test_pt_to_mlx_accepts_numpy_and_scalars()
    test_block_matches_pytorch_reference()
    test_block_with_cross_attn_runs()
    print("OK — all tests passed")
