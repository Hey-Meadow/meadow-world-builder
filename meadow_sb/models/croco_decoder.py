"""MLX port of the CroCo cross-view decoder used by YoNoSplat / Pi3 backbone.

Despite the "cross-view" name, the actual upstream module
(`src.model.encoder.backbone.backbone_local_global.BackboneLocalGlobal`) builds
the decoder as a stack of 36 `BlockRope` modules — **self-attention only with
RoPE2D**. The "cross-view" fusion happens at the *reshape* level inside
`decode()`:

    for i, blk in enumerate(self.decoder):
        if i % 2 == 0:                    # per-view attention
            x   = x.reshape(B*N, hw, C)
            pos = pos.reshape(B*N, hw, 2)
        else:                             # cross-view attention (token-mixed)
            x   = x.reshape(B, N*hw, C)
            pos = pos.reshape(B, N*hw, 2)
        x = blk(x, xpos=pos)

i.e. even blocks see `(B*N, 261, 1024)`, odd blocks see `(B, N*261, 1024)`. The
self-attention's RoPE2D consumes the per-token `(y, x)` position grid, so all
"multi-view" mixing is implicit in how the sequence is concatenated.

Each block has **18 tensors**: norm1, attn.qkv, attn.proj, attn.q_norm,
attn.k_norm, ls1, norm2, mlp.fc1, mlp.fc2, ls2 (×18 weight+bias counts after
counting per-tensor weights and biases). 36 blocks × 18 = 648 tensors,
matching `state_dict_tensor_map.json`.

Numerical reference: `research/yonosplat_bootstrap/dumps/per_block/dec_block_*.npz`
captured by `dump_pi3.py`, which records the *positional* input to each
`BlockRope.forward(x, xpos=pos)` as `in` (positional arg 0) and the residual
output as `out`. `xpos` is a kwarg and was *not* saved; we reconstruct it
deterministically from `PositionGetter` + the prepend-special-tokens rule
(`pos+1`, then 5 zero rows).

Per-block parity (`tests/test_croco_decoder.py`, fp32, 2-view 224x224 input):

    block | max|mlx - pt|     block | max|mlx - pt|
    ------+--------------    ------+--------------
       0  | 2.05e-05            18 | 2.24e-06
       1  | 6.32e-06            19 | 2.38e-06
       2  | 5.60e-06            20 | 4.59e-06
       3  | 7.03e-06            21 | 2.56e-06
       4  | 2.50e-06            22 | 5.96e-06
       5  | 2.86e-06            23 | 6.56e-06
       6  | 2.98e-06            24 | 8.82e-06
       7  | 5.25e-06            25 | 7.87e-06
       8  | 5.48e-06            26 | 8.58e-06
       9  | 3.82e-06            27 | 7.87e-06
      10  | 1.79e-06            28 | 2.37e-05
      11  | 1.67e-06            29 | 1.07e-05
      12  | 1.67e-06            30 | 1.31e-05
      13  | 1.79e-06            31 | 1.65e-05
      14  | 2.62e-06            32 | 1.59e-05
      15  | 3.34e-06            33 | 1.24e-05
      16  | 2.27e-06            34 | 1.91e-05
      17  | 2.21e-06            35 | 1.91e-05

All 36/36 pass the 1e-3 gate by ~50x margin; worst case 2.37e-05 at block 28.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# RoPE2D — pure-MLX port of `src.model.encoder.layers.pos_embed.RoPE2D`
# ---------------------------------------------------------------------------
#
# Reference (PyTorch, slow fallback path):
#
#     class RoPE2D(nn.Module):
#         def __init__(self, freq=100.0, F0=1.0): ...
#         def forward(self, tokens, positions):
#             # tokens: (B, H, N, D), D even
#             # positions: (B, N, 2)
#             y, x = tokens.chunk(2, dim=-1)          # split D into two halves
#             y = apply_rope1d(y, positions[:,:,0])   # rotate by y-position
#             x = apply_rope1d(x, positions[:,:,1])   # rotate by x-position
#             return cat([y, x], dim=-1)
#
# `apply_rope1d`:
#     cos/sin tables come from inv_freq = 1 / freq**(arange(0, D/2, step=2) / (D/2)),
#     where D here is *half of the original head_dim* (we already split).
#     freqs = einsum("i,j->ij", t, inv_freq); cos = cat([freqs, freqs]).cos();
#     same for sin. Then:  out = tokens*cos + rotate_half(tokens)*sin.


def _rope2d_cos_sin(half_D: int, max_pos: int, base: float, dtype) -> Tuple[mx.array, mx.array]:
    """Compute the cos/sin lookup tables for the 1-D RoPE applied on one half.

    Args:
        half_D: feature dim of *one* rope half (= head_dim // 4 in our case,
            because we first split head_dim into (y_half, x_half), then RoPE1D
            takes that half and internally builds inv_freq over `range(0, D, 2)`
            where `D = half_D`).
        max_pos: number of distinct positions to tabulate (= max(positions) + 1).
        base: RoPE frequency base (`freq` argument, == 100 here).
        dtype: target dtype.

    Returns:
        (cos, sin) each of shape (max_pos, half_D).
    """
    # inv_freq: (half_D // 2,)
    inv_freq = 1.0 / (base ** (mx.arange(0, half_D, 2, dtype=mx.float32) / half_D))
    t = mx.arange(max_pos, dtype=mx.float32)
    freqs = t[:, None] * inv_freq[None, :]                  # (max_pos, half_D//2)
    freqs = mx.concatenate([freqs, freqs], axis=-1)         # (max_pos, half_D)
    return mx.cos(freqs).astype(dtype), mx.sin(freqs).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    """Equivalent of PT `rotate_half`: cat([-x2, x1], dim=-1)."""
    D = x.shape[-1]
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope1d(tokens: mx.array, pos1d: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply 1-D rotary position embedding.

    tokens: (B, H, N, half_D)
    pos1d : (B, N) integer index into cos/sin tables
    cos, sin: (max_pos, half_D)
    """
    # Embedding lookup, broadcast across heads.
    # cos[pos1d] -> (B, N, half_D); we want (B, 1, N, half_D) to broadcast over heads.
    cos_emb = cos[pos1d][:, None, :, :]                     # (B, 1, N, half_D)
    sin_emb = sin[pos1d][:, None, :, :]
    return tokens * cos_emb + _rotate_half(tokens) * sin_emb


class RoPE2D:
    """MLX port of `RoPE2D(freq=100.0)`.

    Stateless container — caches cos/sin tables per (half_D, max_pos, dtype).
    Not a `nn.Module` because it has no learnable parameters.
    """

    def __init__(self, freq: float = 100.0):
        self.base = float(freq)
        self._cache: Dict[Tuple[int, int, str], Tuple[mx.array, mx.array]] = {}

    def _get(self, half_D: int, max_pos: int, dtype) -> Tuple[mx.array, mx.array]:
        # dtype as string key (mlx dtypes aren't hashable in older versions).
        key = (half_D, max_pos, str(dtype))
        if key not in self._cache:
            self._cache[key] = _rope2d_cos_sin(half_D, max_pos, self.base, dtype)
        return self._cache[key]

    def __call__(self, tokens: mx.array, positions: mx.array) -> mx.array:
        """Rotate `tokens` by 2-D RoPE.

        Args:
            tokens: (B, H, N, D) — D must be even.
            positions: (B, N, 2) integer (y, x) per token.

        Returns:
            Rotated tokens, same shape as input.
        """
        D = tokens.shape[-1]
        assert D % 2 == 0, "head_dim must be even"
        half_D = D // 2                      # split y/x halves
        # positions is int32/int64 — coerce to int32 for safe indexing.
        if positions.dtype not in (mx.int32, mx.int64):
            positions = positions.astype(mx.int32)
        max_pos = int(positions.max().item()) + 1
        cos, sin = self._get(half_D, max_pos, tokens.dtype)

        y, x = tokens[..., :half_D], tokens[..., half_D:]
        y = _apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = _apply_rope1d(x, positions[:, :, 1], cos, sin)
        return mx.concatenate([y, x], axis=-1)


# ---------------------------------------------------------------------------
# AttentionRope — self-attention with qk_norm + RoPE2D
# ---------------------------------------------------------------------------
#
# Matches `FlashAttentionRope` upstream — same weights as `AttentionRope`.


class AttentionRope(nn.Module):
    """Self-attention with QK-LayerNorm and RoPE2D, matching upstream
    `FlashAttentionRope(qk_norm=True, rope=RoPE2D)`.
    """

    def __init__(self, dim: int, num_heads: int, rope: RoPE2D, qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # qkv: Linear(dim, 3*dim, bias=qkv_bias) — PT weight shape (3*dim, dim).
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        # q_norm / k_norm over head_dim — PT `LayerNorm(head_dim)`.
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.rope = rope

    def __call__(self, x: mx.array, xpos: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x)                                   # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, H, D)                    # (B, N, 3, H, D)
        # (3, B, H, N, D)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # each (B, H, N, D)

        # LayerNorm over last dim (head_dim).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE2D rotates q, k.
        q = self.rope(q, xpos)
        k = self.rope(k, xpos)

        # Scaled dot-product attention.
        # MLX's mx.fast.scaled_dot_product_attention expects (..., N, D) and
        # an explicit scale; it handles the matmul + softmax in one fused kernel.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)    # (B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# LayerScale, MLP, BlockRope
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Learned per-channel scale (`gamma`). Matches PT `LayerScale`."""

    def __init__(self, dim: int, init_values: float = 0.01):
        super().__init__()
        self.gamma = mx.full((dim,), init_values, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class Mlp(nn.Module):
    """Two-layer MLP with GELU (exact, not approx). Matches `dinov2.layers.mlp.Mlp`."""

    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = nn.gelu(x)            # exact GELU (PT `nn.GELU()` default)
        x = self.fc2(x)
        return x


class BlockRope(nn.Module):
    """Single decoder block (self-attn with RoPE + qk-norm + MLP, both wrapped
    in pre-norm + LayerScale).

    State-dict keys (per block):
        norm1.weight, norm1.bias
        attn.qkv.weight, attn.qkv.bias
        attn.proj.weight, attn.proj.bias
        attn.q_norm.weight, attn.q_norm.bias
        attn.k_norm.weight, attn.k_norm.bias
        ls1.gamma
        norm2.weight, norm2.bias
        mlp.fc1.weight, mlp.fc1.bias
        mlp.fc2.weight, mlp.fc2.bias
        ls2.gamma
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, rope: RoPE2D):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = AttentionRope(dim, num_heads, rope=rope)
        self.ls1 = LayerScale(dim, init_values=0.01)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values=0.01)

    def __call__(self, x: mx.array, xpos: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), xpos=xpos))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Position grid helper (mirrors PT `PositionGetter` + the prepend-special-tokens
# branch inside `BackboneLocalGlobal.decode`)
# ---------------------------------------------------------------------------


def build_decoder_positions(B: int, N: int, h: int, w: int, num_register_tokens: int = 5) -> mx.array:
    """Construct the per-token (y, x) position grid passed to BlockRope.

    Mirrors:

        pos = position_getter(B*N, h, w)          # (B*N, h*w, 2), 0-indexed
        pos = pos + 1                              # leave 0 for special tokens
        pos_special = zeros(B*N, num_register, 2)
        pos = concat([pos_special, pos], dim=1)   # (B*N, num_register + h*w, 2)

    Returns shape `(B*N, num_register + h*w, 2)`, int32.
    """
    # y, x = cartesian_prod(arange(h), arange(w)) — y goes slow, x fast.
    y = mx.arange(h, dtype=mx.int32)
    x = mx.arange(w, dtype=mx.int32)
    yy = mx.broadcast_to(y[:, None], (h, w))                # (h, w)
    xx = mx.broadcast_to(x[None, :], (h, w))                # (h, w)
    grid = mx.stack([yy, xx], axis=-1).reshape(1, h * w, 2)  # (1, h*w, 2)
    grid = grid + 1                                          # +1 for special tokens
    grid = mx.broadcast_to(grid, (B * N, h * w, 2))
    special = mx.zeros((B * N, num_register_tokens, 2), dtype=mx.int32)
    return mx.concatenate([special, grid], axis=1)


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------


class CroCoDecoder(nn.Module):
    """36-block CroCo-style cross-view decoder.

    Forward expects already-encoded patch tokens of shape `(B*N, h*w, dim)`
    plus the original B, N, H, W metadata so that:
      * register tokens get prepended
      * the per-block reshape between (B*N, hw, C) and (B, N*hw, C) is applied
      * RoPE2D positions are correct
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        depth: int = 36,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 5,
        rope_freq: float = 100.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.depth = depth
        self.num_register_tokens = num_register_tokens

        self.rope = RoPE2D(freq=rope_freq)
        # 1 × 1 × num_register × dim — registered as a parameter at the
        # `encoder.backbone.register_token` key in the upstream state-dict.
        self.register_token = mx.zeros((1, 1, num_register_tokens, dim), dtype=mx.float32)

        self.blocks = [
            BlockRope(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, rope=self.rope)
            for _ in range(depth)
        ]

    def __call__(
        self,
        hidden: mx.array,
        B: int,
        N: int,
        H: int,
        W: int,
        patch_size: int = 14,
    ) -> Tuple[mx.array, mx.array]:
        """Run the full 36-block decoder.

        Args:
            hidden: (B*N, h*w, dim) patch tokens from the encoder (intrinsic
                embedding already added). h = H // patch_size etc.
            B, N, H, W: batch / view / spatial dims of the original image.

        Returns:
            (out, pos) where:
              out: (B*N, num_register + h*w, dim) the final hidden state
              pos: same shape, the final position grid (matches PT contract).
        """
        h, w = H // patch_size, W // patch_size
        hw = self.num_register_tokens + h * w

        # Prepend register tokens.
        reg = mx.broadcast_to(self.register_token, (B, N, self.num_register_tokens, self.dim))
        reg = reg.reshape(B * N, self.num_register_tokens, self.dim)
        hidden = mx.concatenate([reg, hidden], axis=1)      # (B*N, hw, dim)

        pos = build_decoder_positions(B, N, h, w, self.num_register_tokens)

        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                hidden = hidden.reshape(B * N, hw, self.dim)
                pos_blk = pos.reshape(B * N, hw, 2)
            else:
                hidden = hidden.reshape(B, N * hw, self.dim)
                pos_blk = pos.reshape(B, N * hw, 2)
            hidden = blk(hidden, xpos=pos_blk)

        # Final reshape back to (B*N, hw, dim) for downstream sub-decoders.
        hidden = hidden.reshape(B * N, hw, self.dim)
        pos = pos.reshape(B * N, hw, 2)
        return hidden, pos


# ---------------------------------------------------------------------------
# Weight loading from the PT state-dict slice
# ---------------------------------------------------------------------------


def _np_to_mx(a: np.ndarray) -> mx.array:
    return mx.array(a)


def load_block_weights_from_pt(block: BlockRope, pt_sd: Dict[str, np.ndarray], prefix: str) -> None:
    """Load one BlockRope from the PT state-dict slice `pt_sd` with the given
    `prefix` (e.g. `"encoder.backbone.decoder.0."`).

    Linear layout matches between PT (`(out, in)`) and MLX (`(out, in)`); no
    transpose required.
    """

    def get(key: str) -> mx.array:
        return _np_to_mx(pt_sd[prefix + key])

    # norm1
    block.norm1.weight = get("norm1.weight")
    block.norm1.bias = get("norm1.bias")
    # attn
    block.attn.qkv.weight = get("attn.qkv.weight")
    block.attn.qkv.bias = get("attn.qkv.bias")
    block.attn.proj.weight = get("attn.proj.weight")
    block.attn.proj.bias = get("attn.proj.bias")
    block.attn.q_norm.weight = get("attn.q_norm.weight")
    block.attn.q_norm.bias = get("attn.q_norm.bias")
    block.attn.k_norm.weight = get("attn.k_norm.weight")
    block.attn.k_norm.bias = get("attn.k_norm.bias")
    # ls1
    block.ls1.gamma = get("ls1.gamma")
    # norm2
    block.norm2.weight = get("norm2.weight")
    block.norm2.bias = get("norm2.bias")
    # mlp
    block.mlp.fc1.weight = get("mlp.fc1.weight")
    block.mlp.fc1.bias = get("mlp.fc1.bias")
    block.mlp.fc2.weight = get("mlp.fc2.weight")
    block.mlp.fc2.bias = get("mlp.fc2.bias")
    # ls2
    block.ls2.gamma = get("ls2.gamma")


def load_decoder_weights_from_pt(decoder: CroCoDecoder, pt_sd: Dict[str, np.ndarray]) -> None:
    """Load all 36 blocks + register_token from the PT state-dict slice."""
    decoder.register_token = _np_to_mx(pt_sd["encoder.backbone.register_token"])
    for i, blk in enumerate(decoder.blocks):
        load_block_weights_from_pt(blk, pt_sd, f"encoder.backbone.decoder.{i}.")


__all__ = [
    "RoPE2D",
    "AttentionRope",
    "LayerScale",
    "Mlp",
    "BlockRope",
    "CroCoDecoder",
    "build_decoder_positions",
    "load_block_weights_from_pt",
    "load_decoder_weights_from_pt",
]
