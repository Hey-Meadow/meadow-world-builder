"""MLX port of the three YoNoSplat sub-decoders (point / gaussian / camera).

Each sub-decoder is a `TransformerDecoder` from
`src.model.encoder.layers.transformer_head` — i.e. a stack of
**self-attention** `BlockRope` modules (no cross-attention). Wrapping the
stack is a per-decoder `projects` Linear (2*dec_embed_dim → dec_embed_dim)
and `linear_out` Linear (dec_embed_dim → out_dim).

Differences from the CroCo backbone decoder (Agent B's port):
  * **depth = 5** (not 36) — confirmed by state-dict (5 blocks × 12 tensors
    + 4 wrapping tensors = 64 tensors per sub-decoder).
  * **qk_norm = False** — no `attn.q_norm` / `attn.k_norm` LayerNorms.
  * **init_values = None** — no `LayerScale`; the upstream uses
    `nn.Identity()` for `ls1`/`ls2`.
  * Top-level `projects` (in_dim → dec_embed_dim) precedes the block stack,
    and a final `linear_out` (dec_embed_dim → out_dim) follows it.

Tensor inventory (per sub-decoder, prefix `encoder.{point,gaussian,camera}_decoder.`):

    projects.weight                   (1024, 2048)
    projects.bias                     (1024,)
    blocks.0..4.norm1.weight          (1024,)
    blocks.0..4.norm1.bias            (1024,)
    blocks.0..4.attn.qkv.weight       (3072, 1024)
    blocks.0..4.attn.qkv.bias         (3072,)
    blocks.0..4.attn.proj.weight      (1024, 1024)
    blocks.0..4.attn.proj.bias        (1024,)
    blocks.0..4.norm2.weight          (1024,)
    blocks.0..4.norm2.bias            (1024,)
    blocks.0..4.mlp.fc1.weight        (4096, 1024)
    blocks.0..4.mlp.fc1.bias          (4096,)
    blocks.0..4.mlp.fc2.weight        (1024, 4096)
    blocks.0..4.mlp.fc2.bias          (1024,)
    linear_out.weight                 (out_dim, 1024)   # out_dim = 1024 for
                                                        # point/gaussian, 512 for camera
    linear_out.bias                   (out_dim,)

That sums to 12 per block × 5 blocks + 4 wrapping = 64 tensors. The
state-dict on disk matches exactly (`grep -c 'encoder.point_decoder'
state_dict_tensor_map.json` → 64).

Forward signature is identical to upstream:

    out = sub_decoder(hidden, xpos=pos)
    # hidden: (B*V, N, 2048)
    # pos:    (B*V, N, 2) int  — RoPE2D positions
    # out:    (B*V, N, out_dim)

RoPE2D is reused from `meadow_sb.models.croco_decoder` so the cos/sin
cache is shared across all sub-decoders and the backbone (matching
PT's `self.backbone.rope` sharing pattern).
"""
from __future__ import annotations

from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# RoPE2D — inlined here (rather than importing from croco_decoder) to keep
# this module self-contained for parallel agent landing. Once Agent B's
# croco_decoder lands we can switch to a shared import; until then we mirror
# the same math so weight sharing of `rope` between modules is just an
# instance argument.
# ---------------------------------------------------------------------------


def _rope2d_cos_sin(half_D: int, max_pos: int, base: float, dtype):
    inv_freq = 1.0 / (base ** (mx.arange(0, half_D, 2, dtype=mx.float32) / half_D))
    t = mx.arange(max_pos, dtype=mx.float32)
    freqs = t[:, None] * inv_freq[None, :]
    freqs = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(freqs).astype(dtype), mx.sin(freqs).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    D = x.shape[-1]
    return mx.concatenate([-x[..., D // 2 :], x[..., : D // 2]], axis=-1)


def _apply_rope1d(tokens: mx.array, pos1d: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    cos_emb = cos[pos1d][:, None, :, :]
    sin_emb = sin[pos1d][:, None, :, :]
    return tokens * cos_emb + _rotate_half(tokens) * sin_emb


class RoPE2D:
    """Stateless 2-D RoPE container with a cos/sin cache.

    Mirrors `src.model.encoder.layers.pos_embed.RoPE2D` (slow PT fallback)
    and Agent B's `meadow_sb.models.croco_decoder.RoPE2D`.
    """

    def __init__(self, freq: float = 100.0):
        self.base = float(freq)
        self._cache: Dict = {}

    def _get(self, half_D: int, max_pos: int, dtype):
        key = (half_D, max_pos, str(dtype))
        if key not in self._cache:
            self._cache[key] = _rope2d_cos_sin(half_D, max_pos, self.base, dtype)
        return self._cache[key]

    def __call__(self, tokens: mx.array, positions: mx.array) -> mx.array:
        D = tokens.shape[-1]
        assert D % 2 == 0, "head_dim must be even"
        half_D = D // 2
        if positions.dtype not in (mx.int32, mx.int64):
            positions = positions.astype(mx.int32)
        max_pos = int(positions.max().item()) + 1
        cos, sin = self._get(half_D, max_pos, tokens.dtype)
        y, x = tokens[..., :half_D], tokens[..., half_D:]
        y = _apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = _apply_rope1d(x, positions[:, :, 1], cos, sin)
        return mx.concatenate([y, x], axis=-1)


# ---------------------------------------------------------------------------
# AttentionRope (no qk_norm) and BlockRope (no LayerScale) — sub-decoder flavour
# ---------------------------------------------------------------------------


class AttentionRope(nn.Module):
    """Self-attention with RoPE2D but **no** q/k LayerNorm.

    Matches `FlashAttentionRope(qk_norm=False, rope=RoPE2D)` upstream — the
    `q_norm`/`k_norm` modules are `nn.Identity` and not present in the
    state-dict.
    """

    def __init__(self, dim: int, num_heads: int, rope: RoPE2D,
                 qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope

    def __call__(self, x: mx.array, xpos: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each (B, H, N, D)

        q = self.rope(q, xpos)
        k = self.rope(k, xpos)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    """Two-layer MLP with exact GELU. Matches `dinov2.layers.mlp.Mlp`."""

    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class BlockRope(nn.Module):
    """Self-attn + MLP block with pre-norm residual (no LayerScale).

    State-dict keys (per block, prefix `blocks.<i>.`):
        norm1.weight, norm1.bias
        attn.qkv.weight, attn.qkv.bias
        attn.proj.weight, attn.proj.bias
        norm2.weight, norm2.bias
        mlp.fc1.weight, mlp.fc1.bias
        mlp.fc2.weight, mlp.fc2.bias
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, rope: RoPE2D):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = AttentionRope(dim, num_heads, rope=rope)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def __call__(self, x: mx.array, xpos: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), xpos=xpos)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Sub-decoder base + three named subclasses
# ---------------------------------------------------------------------------


class TransformerSubDecoder(nn.Module):
    """Generic transformer sub-decoder shared by point / gaussian / camera.

    Layout:
        projects  : Linear(in_dim, dec_embed_dim)
        blocks    : [BlockRope] * depth
        linear_out: Linear(dec_embed_dim, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 2048,
        out_dim: int = 1024,
        dec_embed_dim: int = 1024,
        depth: int = 5,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        rope: Optional[RoPE2D] = None,
        rope_freq: float = 100.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dec_embed_dim = dec_embed_dim
        self.depth = depth

        self.rope = rope if rope is not None else RoPE2D(freq=rope_freq)
        self.projects = nn.Linear(in_dim, dec_embed_dim)
        self.blocks = [
            BlockRope(dec_embed_dim, num_heads, mlp_ratio, self.rope)
            for _ in range(depth)
        ]
        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def __call__(self, hidden: mx.array, xpos: mx.array) -> mx.array:
        h = self.projects(hidden)
        for blk in self.blocks:
            h = blk(h, xpos=xpos)
        return self.linear_out(h)


class PointDecoder(TransformerSubDecoder):
    """5-block transformer that consumes backbone tokens (B*V, N, 2048) and
    emits point-aware tokens (B*V, N, 1024). Fed into `LinearPts3d` head
    that produces per-token 3-D points after pixel-shuffle upsampling."""

    def __init__(self, rope: Optional[RoPE2D] = None, **kwargs):
        super().__init__(in_dim=2048, out_dim=1024, dec_embed_dim=1024,
                         depth=5, num_heads=16, mlp_ratio=4.0, rope=rope, **kwargs)


class GaussianDecoder(TransformerSubDecoder):
    """5-block transformer for Gaussian feature regression. Same shape as
    PointDecoder; upstream constructs it via `deepcopy(point_decoder)` so the
    architecture is identical — only the trained weights differ."""

    def __init__(self, rope: Optional[RoPE2D] = None, **kwargs):
        super().__init__(in_dim=2048, out_dim=1024, dec_embed_dim=1024,
                         depth=5, num_heads=16, mlp_ratio=4.0, rope=rope, **kwargs)


class CameraDecoder(TransformerSubDecoder):
    """5-block transformer for camera-pose tokens. **out_dim is 512**, not
    1024 — the linear_out projects down to the CameraHead's input width."""

    def __init__(self, rope: Optional[RoPE2D] = None, **kwargs):
        super().__init__(in_dim=2048, out_dim=512, dec_embed_dim=1024,
                         depth=5, num_heads=16, mlp_ratio=4.0, rope=rope, **kwargs)


# ---------------------------------------------------------------------------
# Weight loading from the PT state-dict slice
# ---------------------------------------------------------------------------


def _np_to_mx(a: np.ndarray) -> mx.array:
    return mx.array(a)


def load_block_weights(block: BlockRope, pt_sd: Dict[str, np.ndarray], prefix: str) -> None:
    def get(key: str) -> mx.array:
        return _np_to_mx(pt_sd[prefix + key])

    block.norm1.weight = get("norm1.weight")
    block.norm1.bias = get("norm1.bias")
    block.attn.qkv.weight = get("attn.qkv.weight")
    block.attn.qkv.bias = get("attn.qkv.bias")
    block.attn.proj.weight = get("attn.proj.weight")
    block.attn.proj.bias = get("attn.proj.bias")
    block.norm2.weight = get("norm2.weight")
    block.norm2.bias = get("norm2.bias")
    block.mlp.fc1.weight = get("mlp.fc1.weight")
    block.mlp.fc1.bias = get("mlp.fc1.bias")
    block.mlp.fc2.weight = get("mlp.fc2.weight")
    block.mlp.fc2.bias = get("mlp.fc2.bias")


def load_sub_decoder_weights(decoder: TransformerSubDecoder,
                             pt_sd: Dict[str, np.ndarray],
                             prefix: str) -> None:
    """Load one sub-decoder from `pt_sd` keyed by `prefix` (with trailing dot),
    e.g. `'encoder.point_decoder.'`.
    """
    decoder.projects.weight = _np_to_mx(pt_sd[prefix + "projects.weight"])
    decoder.projects.bias = _np_to_mx(pt_sd[prefix + "projects.bias"])
    for i, blk in enumerate(decoder.blocks):
        load_block_weights(blk, pt_sd, f"{prefix}blocks.{i}.")
    decoder.linear_out.weight = _np_to_mx(pt_sd[prefix + "linear_out.weight"])
    decoder.linear_out.bias = _np_to_mx(pt_sd[prefix + "linear_out.bias"])


__all__ = [
    "RoPE2D",
    "AttentionRope",
    "Mlp",
    "BlockRope",
    "TransformerSubDecoder",
    "PointDecoder",
    "GaussianDecoder",
    "CameraDecoder",
    "load_sub_decoder_weights",
    "load_block_weights",
]
