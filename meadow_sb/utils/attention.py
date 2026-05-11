"""MLX building blocks for the YoNoSplat port (DINOv2-style ViT primitives).

Layouts match `timm.layers.Attention` / `timm.models.vision_transformer.Block`
and `timm.layers.LayerScale`, which is what DINOv2 / CroCo derivatives use.

PT parameter names this file reproduces (so weight loading is trivial):

* ``norm1.weight / norm1.bias``           — LayerNorm before attention
* ``attn.qkv.weight / attn.qkv.bias``     — fused QKV projection
* ``attn.proj.weight / attn.proj.bias``   — output projection
* ``ls1.gamma``                           — LayerScale after attention
* ``norm2.weight / norm2.bias``           — LayerNorm before MLP
* ``mlp.fc1.weight / mlp.fc1.bias``       — MLP up
* ``mlp.fc2.weight / mlp.fc2.bias``       — MLP down
* ``ls2.gamma``                           — LayerScale after MLP

Cross-attention variant adds:

* ``cross_attn.q.weight / .bias``         — query proj (from x)
* ``cross_attn.kv.weight / .bias``        — key+value proj (from context)
* ``cross_attn.proj.weight / .bias``      — output proj
* ``norm_y.weight / norm_y.bias``         — LayerNorm on context (CroCo style)

RoPE is supplied externally as a per-call callable
``rope_fn(q, k) -> (q_rot, k_rot)`` to keep this module decoupled from the
specific RoPE implementation the encoder/decoder agents pick.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


RopeFn = Callable[[mx.array, mx.array], Tuple[mx.array, mx.array]]


# ---------------------------------------------------------------------------
# Attention (self)
# ---------------------------------------------------------------------------


class MLXAttention(nn.Module):
    """Standard ViT multi-head self-attention with fused QKV projection.

    Forward: ``x`` shape ``(B, N, C)`` -> ``(B, N, C)``.
    Uses :func:`mx.fast.scaled_dot_product_attention` (Metal-fused on Apple
    Silicon). RoPE, if needed, is applied to ``q,k`` via ``rope_fn`` before
    the dot product (caller supplies the actual RoPE module).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        if qk_norm:
            # LayerNorm per-head dim, matching timm's `q_norm`/`k_norm` Identity-or-LN.
            self.q_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        rope_fn: Optional[RopeFn] = None,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x)
        # (B, N, 3, H, D) -> split q/k/v as (B, N, H, D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (B, H, N, D) for SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if rope_fn is not None:
            q, k = rope_fn(q, k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attn_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Cross-attention (Q from x, K/V from context)
# ---------------------------------------------------------------------------


class MLXCrossAttention(nn.Module):
    """Cross-attention block. ``q = Wq(x)``, ``k,v = Wkv(context)``.

    Used by the CroCo-style decoder (Agent B). Mirrors the DUSt3R/CroCo
    convention: separate ``q`` linear and packed ``kv`` linear.
    """

    def __init__(
        self,
        dim: int,
        ctx_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        ctx_dim = ctx_dim or dim
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, N, C = x.shape
        M = context.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0].transpose(0, 2, 1, 3)
        v = kv[:, :, 1].transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attn_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# LayerScale (DINOv2)
# ---------------------------------------------------------------------------


class MLXLayerScale(nn.Module):
    """Per-channel learnable scale (DINOv2). Param name matches timm: ``gamma``."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = mx.full((dim,), init_values, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLXMlp(nn.Module):
    """Standard ViT MLP: Linear -> GELU -> Linear. Param names ``fc1`` / ``fc2``."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
        act: str = "gelu",
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        if act == "gelu":
            self._act = nn.gelu
        elif act == "gelu_approx" or act == "gelu_tanh":
            self._act = nn.gelu_approx
        else:
            raise ValueError(f"unsupported act={act!r}")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self._act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Block (self + optional cross + MLP)
# ---------------------------------------------------------------------------


class MLXBlock(nn.Module):
    """ViT block with DINOv2 LayerScale residuals; optional cross-attention.

    Pattern (DINOv2)::

        x = x + ls1(attn(norm1(x)))
        x = x + ls2(mlp(norm2(x)))

    With cross-attention enabled (CroCo / dust3r decoder)::

        x = x + ls1(attn(norm1(x)))
        x = x + ls_cross(cross_attn(norm_xq(x), norm_y(y)))
        x = x + ls2(mlp(norm2(x)))

    LayerScale modules are always instantiated when ``init_values`` is set; if
    ``init_values is None`` they're replaced by identity (param-less) ops so the
    block is bit-compatible with non-DINOv2 ViTs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = 1e-5,
        norm_eps: float = 1e-6,
        act: str = "gelu",
        use_cross_attn: bool = False,
        ctx_dim: Optional[int] = None,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.has_ls = init_values is not None

        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = MLXAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, proj_bias=proj_bias,
            qk_norm=qk_norm, norm_eps=norm_eps,
        )
        if self.has_ls:
            self.ls1 = MLXLayerScale(dim, init_values=init_values)

        if use_cross_attn:
            # CroCo names the pre-norms ``norm_xq`` (query side) and ``norm_y``
            # (context side). We expose both as attributes so a weight loader
            # can map directly.
            self.norm_xq = nn.LayerNorm(dim, eps=norm_eps)
            self.norm_y = nn.LayerNorm((ctx_dim or dim), eps=norm_eps)
            self.cross_attn = MLXCrossAttention(
                dim, ctx_dim=ctx_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, proj_bias=proj_bias,
            )
            if self.has_ls:
                self.ls_cross = MLXLayerScale(dim, init_values=init_values)

        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLXMlp(
            dim,
            hidden_features=int(dim * mlp_ratio),
            bias=proj_bias,
            act=act,
        )
        if self.has_ls:
            self.ls2 = MLXLayerScale(dim, init_values=init_values)

    def __call__(
        self,
        x: mx.array,
        context: Optional[mx.array] = None,
        rope_fn: Optional[RopeFn] = None,
        rope_fn_cross: Optional[RopeFn] = None,
        attn_mask: Optional[mx.array] = None,
        cross_attn_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention sublayer
        h = self.attn(self.norm1(x), rope_fn=rope_fn, attn_mask=attn_mask)
        if self.has_ls:
            h = self.ls1(h)
        x = x + h

        # Optional cross-attention sublayer
        if self.use_cross_attn:
            if context is None:
                raise ValueError("MLXBlock(use_cross_attn=True) requires context")
            # CroCo applies separate norms to query (from x) and context (from y).
            h = self.cross_attn(
                self.norm_xq(x),
                self.norm_y(context),
                attn_mask=cross_attn_mask,
            )
            if self.has_ls:
                h = self.ls_cross(h)
            x = x + h

        # MLP sublayer
        h = self.mlp(self.norm2(x))
        if self.has_ls:
            h = self.ls2(h)
        x = x + h
        return x


__all__ = [
    "MLXAttention",
    "MLXCrossAttention",
    "MLXLayerScale",
    "MLXMlp",
    "MLXBlock",
    "RopeFn",
]
