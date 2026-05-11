"""MLX port of SAM 3D Objects DiT (Diffusion Transformer) backbone.

Mirrors `sam3d_objects/model/backbone/tdfy_dit/`.

Two architecture variants live in this file:

1. ``DiTBlock`` / ``DiTCrossBlock``  — single-modality DiT blocks. These mirror
   PT ``ModulatedTransformerBlock`` / ``ModulatedTransformerCrossBlock``.
2. ``MOTDiTCrossBlock``              — multi-modality (per-latent ModuleDict)
   variant mirroring ``MOTModulatedTransformerCrossBlock`` used by the
   stage-1 ``ss_flow`` checkpoint.

DENSE only — sparse 3D ops (octree, spconv) are handled by OBJ-METAL-SPARSE.

Conventions
-----------
- Sequence layout: ``(B, N, C)``.
- Attention: heads-first ``(B, H, N, D)`` for ``mx.fast.scaled_dot_product_attention``.
- LayerNorm32 in PT casts to fp32 internally; MLX nn.LayerNorm matches for fp32 inputs.
- Both models use ``GELU(approximate='tanh')`` in PT FeedForwardNet → mirror with
  ``nn.gelu_approx`` (tanh-based) in MLX.
- Linear weights: PT ``(out, in)`` == MLX ``(out, in)`` (no transpose needed).

Weight prefix in npz files (``weights/sam3d_objects/{ss,slat}_flow.npz``):
    ``reverse_fn.backbone.``
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gelu_tanh(x: mx.array) -> mx.array:
    """GELU with tanh approximation (matches PT ``nn.GELU(approximate='tanh')``)."""
    # Use MLX's approximate GELU which uses tanh
    return nn.gelu_approx(x)


# ---------------------------------------------------------------------------
# bf16 mixed-precision helpers
# ---------------------------------------------------------------------------
#
# PT inference runs the DiT backbone under ``torch.autocast(dtype=bfloat16)``
# (see ``sam3d_objects/pipeline/inference_pipeline.py:71, 674``). The
# checkpoint is also bf16-trained, so casting weights / activations to bf16
# inside the transformer blocks gives a ~1.5-2x speedup on memory-bound DiT
# stacks (24 blocks * 4096 tokens * 1024 channels) without measurable
# numerical drift.
#
# Strategy:
#   * Block-internal weights (Linear, RMSNorm gamma) -> bf16
#   * LayerNorm gamma/beta inside the block also bf16 (input is already bf16
#     by then, so MLX would promote to fp32 anyway; bf16 LN is fine for
#     inference)
#   * Block forward casts input to bf16, returns bf16; backbone wraps
#     fp32 -> bf16 at entry, bf16 -> fp32 at exit
#   * Sampler latents stay fp32 (Euler integrator accumulates in fp32)
#   * AdaLN modulation is computed in bf16 inside the block (cond comes
#     from fp32 t_embedder, cast to bf16 once before the per-block
#     adaLN_modulation)
#   * Cross-attention context (image features) stays fp32 in the embedder
#     output and is cast to bf16 inside each block on demand
#   * Latent_mapping projections, embedders, GS decoder all stay fp32


def _cast_module_params_to_dtype(module: nn.Module, dtype) -> None:
    """Recursively cast every leaf ``mx.array`` parameter in ``module`` to ``dtype``.

    Uses ``mlx.utils.tree_map`` over the parameter tree, then ``module.update``
    to write the cast tensors back. Handles nested dicts/lists of submodules
    (used by ``MOTDiTCrossBlock`` per-modality ModuleDicts) automatically.
    """
    from mlx.utils import tree_map

    def _cast(x):
        if isinstance(x, mx.array) and x.dtype != dtype:
            return x.astype(dtype)
        return x

    new_params = tree_map(_cast, module.parameters())
    module.update(new_params)


# ---------------------------------------------------------------------------
# Embedders / norms
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> Linear -> SiLU -> Linear (matches PT)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        # PT mlp = Sequential(Linear, SiLU, Linear); we keep mlp.0 / mlp.2 names
        # via list-of-layers attribute "mlp" with .0 and .2 lookups in loader.
        self.fc0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def _timestep_embedding(self, t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period)
            * mx.arange(0, half, dtype=mx.float32)
            / half
        )
        if t.ndim == 0:
            t = t[None]
        args = t.astype(mx.float32)[:, None] * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        emb = self._timestep_embedding(t, self.frequency_embedding_size)
        emb = self.fc0(emb)
        emb = nn.silu(emb)
        emb = self.fc2(emb)
        return emb


class MultiHeadRMSNorm(nn.Module):
    """Per-head L2-normalize last dim, scale by gamma * sqrt(dim).

    Matches PT ``modules/attention/modules.py::MultiHeadRMSNorm``::

        out = F.normalize(x, dim=-1) * gamma * sqrt(dim)

    ``gamma`` shape: (heads, dim)
    Input shape:    (B, N, H, D) or (B, H, N, D)  (last dim is normalized;
                    gamma broadcast handles either layout when reshaped).
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x: mx.array) -> mx.array:
        # Normalize over last dim
        norm = mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)
        x = x * norm
        # gamma is (H, D); broadcast — caller ensures axis -2 is H.
        return x * self.gamma * self.scale


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------


class RoPE(nn.Module):
    """Rotary position embedding mirroring PT ``RotaryPositionEmbedder``.

    For PT shape (..., N, D) this rotates pairs ``(x_{2i}, x_{2i+1})`` by
    angle ``indices * freq_i`` with ``freq_i = 1/10000^(i/freq_dim)``.

    PT supports multi-dim coordinates (in_channels=3 for spatial). For the
    common SAM-3D-Objects DiT use this is in_channels=1 (sequence index).
    Implementation here covers the in_channels=1 case (extend if needed).
    """

    def __init__(self, hidden_size: int, in_channels: int = 1):
        super().__init__()
        assert hidden_size % 2 == 0
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        freqs = mx.arange(self.freq_dim, dtype=mx.float32) / self.freq_dim
        self.freqs = 1.0 / (10000.0 ** freqs)

    def _phases(self, indices: mx.array) -> Tuple[mx.array, mx.array]:
        """Return (cos, sin) of indices ⊗ freqs, padded to hidden_size//2."""
        idx_flat = indices.reshape(-1).astype(mx.float32)  # (N,)
        # outer product -> (N, freq_dim)
        ang = idx_flat[:, None] * self.freqs[None, :]
        cos = mx.cos(ang)
        sin = mx.sin(ang)
        target = self.hidden_size // 2
        if cos.shape[1] < target:
            pad = target - cos.shape[1]
            cos = mx.concatenate([cos, mx.ones((cos.shape[0], pad))], axis=1)
            sin = mx.concatenate([sin, mx.zeros((sin.shape[0], pad))], axis=1)
        return cos, sin

    def _rotate(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply rotation. x shape (..., N, D). cos/sin shape (N, D//2)."""
        shape = x.shape
        D = shape[-1]
        x_pair = x.reshape(*shape[:-1], D // 2, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        # broadcast cos/sin over leading dims
        # cos: (N, D//2) -> (..., N, D//2)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = mx.stack([out_even, out_odd], axis=-1)
        return out.reshape(*shape)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """q, k shape: (..., N, D). Returns (q_rot, k_rot)."""
        N = q.shape[-2]
        if indices is None:
            indices = mx.arange(N)
        cos, sin = self._phases(indices)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class FeedForwardNet(nn.Module):
    """Linear -> GELU(tanh) -> Linear. Matches PT ``FeedForwardNet``.

    PT names sub-layers ``mlp.0`` (Linear), ``mlp.1`` (GELU, no params),
    ``mlp.2`` (Linear). We mirror with ``fc0`` / ``fc2`` and a custom loader.
    """

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.fc0 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_gelu_tanh(self.fc0(x)))


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with optional QK-RMS-Norm and RoPE."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels, bias=True)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        if use_rope:
            self.rope = RoPE(channels)

    def __call__(self, x: mx.array, indices: Optional[mx.array] = None) -> mx.array:
        B, N, C = x.shape
        qkv = self.to_qkv(x)
        # (B, N, 3, H, D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # split: (B, N, H, D) for q, k, v
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        if self.use_rope:
            # RoPE expects (..., N, D); apply per-head: shape (B, N, H, D) ->
            # treat (B, H) as leading dims by transposing
            q_t = q.transpose(0, 2, 1, 3)  # (B, H, N, D)
            k_t = k.transpose(0, 2, 1, 3)
            q_t, k_t = self.rope(q_t, k_t, indices)
            q = q_t.transpose(0, 2, 1, 3)
            k = k_t.transpose(0, 2, 1, 3)

        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        # to (B, H, N, D) for SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # (B, H, N, D) -> (B, N, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: Q from x, K/V from context. No RoPE (matches PT)."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_rms_norm = qk_rms_norm

        self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.to_kv = nn.Linear(ctx_channels, channels * 2, bias=qkv_bias)
        self.to_out = nn.Linear(channels, channels, bias=True)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        B, L, C = x.shape
        Lk = context.shape[1]
        q = self.to_q(x).reshape(B, L, self.num_heads, self.head_dim)
        kv = self.to_kv(context).reshape(B, Lk, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0]
        v = kv[:, :, 1]

        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        q = q.transpose(0, 2, 1, 3)  # (B, H, L, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, C)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# AdaLN modulation
# ---------------------------------------------------------------------------


class AdaLNModulation(nn.Module):
    """SiLU -> Linear(c, 6c) producing (shift, scale, gate) for both attn & mlp.

    PT names: ``adaLN_modulation.0`` is SiLU (no params), ``.1`` is Linear.
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 6, bias=True)

    def __call__(self, cond: mx.array) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        h = nn.silu(cond)
        # PT does mod.chunk(6, dim=1) → 6 chunks of size hidden along dim=1
        out = self.proj(h)
        # Each chunk (B, hidden_dim)
        c = out.shape[-1] // 6
        shift_msa = out[:, 0 * c : 1 * c]
        scale_msa = out[:, 1 * c : 2 * c]
        gate_msa = out[:, 2 * c : 3 * c]
        shift_mlp = out[:, 3 * c : 4 * c]
        scale_mlp = out[:, 4 * c : 5 * c]
        gate_mlp = out[:, 5 * c : 6 * c]
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ---------------------------------------------------------------------------
# DiT blocks (single-modality)
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """Modulated DiT block (self-attn + MLP) — mirrors ``ModulatedTransformerBlock``.

    Pattern::

        shift, scale, gate = adaLN(cond).chunk(6)
        h = norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * attn(h)
        h = norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * mlp(h)
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        share_mod: bool = False,
    ):
        super().__init__()
        # PT uses elementwise_affine=False for both norms in ModulatedTransformerBlock
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.attn = MultiHeadSelfAttention(
            channels, num_heads, qkv_bias=qkv_bias,
            use_rope=use_rope, qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.share_mod = share_mod
        if not share_mod:
            assert cond_dim is not None
            self.adaLN_modulation = AdaLNModulation(cond_dim, channels)

    def _split_mod(self, mod: mx.array):
        c = mod.shape[-1] // 6
        return tuple(mod[:, i * c : (i + 1) * c] for i in range(6))

    def __call__(self, x: mx.array, mod: mx.array, indices: Optional[mx.array] = None) -> mx.array:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_mod(mod)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod)

        h = self.norm1(x)
        h = h * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h = self.attn(h, indices=indices)
        h = h * gate_msa[:, None, :]
        x = x + h

        h = self.norm2(x)
        h = h * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = self.mlp(h)
        h = h * gate_mlp[:, None, :]
        x = x + h
        return x


class DiTCrossBlock(nn.Module):
    """Modulated cross-attention DiT block — mirrors ``ModulatedTransformerCrossBlock``.

    Pattern::

        shift, scale, gate = adaLN(cond).chunk(6)
        h = norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self_attn(h)
        h = norm2(x)                          # affine
        x = x + cross_attn(h, context)
        h = norm3(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * mlp(h)

    NOTE: PT ``norm2`` uses ``elementwise_affine=True`` (the only affine norm).
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        share_mod: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, affine=True)
        self.norm3 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.self_attn = MultiHeadSelfAttention(
            channels, num_heads, qkv_bias=qkv_bias,
            use_rope=use_rope, qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadCrossAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.share_mod = share_mod
        if not share_mod:
            assert cond_dim is not None
            self.adaLN_modulation = AdaLNModulation(cond_dim, channels)

    def _split_mod(self, mod: mx.array):
        c = mod.shape[-1] // 6
        return tuple(mod[:, i * c : (i + 1) * c] for i in range(6))

    def __call__(
        self,
        x: mx.array,
        mod: mx.array,
        context: mx.array,
        indices: Optional[mx.array] = None,
    ) -> mx.array:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_mod(mod)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod)

        h = self.norm1(x)
        h = h * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h = self.self_attn(h, indices=indices)
        h = h * gate_msa[:, None, :]
        x = x + h

        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        h = self.norm3(x)
        h = h * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = self.mlp(h)
        h = h * gate_mlp[:, None, :]
        x = x + h
        return x


# ---------------------------------------------------------------------------
# MOT (multi-modality) DiT cross block
# ---------------------------------------------------------------------------


class MOTDiTCrossBlock(nn.Module):
    """Multi-modality cross-attention DiT block.

    Mirrors ``MOTModulatedTransformerCrossBlock``: per-latent ModuleDict for
    norms / cross_attn / mlp / self-attn projections, with shared adaLN.

    Self-attention uses concat-along-tokens with a "protected modality" routing
    pattern: protected modalities (e.g. ``shape``) attend ONLY to themselves;
    other modalities attend to themselves + protected (with K/V detached at
    train time; at inference there's no grad so it's just plain concat-attend).

    Inputs are dicts keyed by latent name. ``context`` is a single tensor
    (image features), shared across all modalities.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        latent_names: List[str],
        cond_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        share_mod: bool = False,
        protect_modality_list: Optional[List[str]] = None,
    ):
        super().__init__()
        if protect_modality_list is None:
            protect_modality_list = ["shape"]
        self.latent_names = list(latent_names)
        self.protect_modality_list = list(protect_modality_list)

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        # ModuleDict-equivalents stored as plain dicts; MLX nn.Module recurses.
        self.norm1: Dict[str, nn.LayerNorm] = {
            n: nn.LayerNorm(channels, eps=1e-6, affine=False) for n in self.latent_names
        }
        self.norm2: Dict[str, nn.LayerNorm] = {
            n: nn.LayerNorm(channels, eps=1e-6, affine=True) for n in self.latent_names
        }
        self.norm3: Dict[str, nn.LayerNorm] = {
            n: nn.LayerNorm(channels, eps=1e-6, affine=False) for n in self.latent_names
        }

        # Self-attn per-modality projections (matches MOTMultiHeadSelfAttention)
        self.sa_to_qkv: Dict[str, nn.Linear] = {
            n: nn.Linear(channels, channels * 3, bias=qkv_bias) for n in self.latent_names
        }
        self.sa_to_out: Dict[str, nn.Linear] = {
            n: nn.Linear(channels, channels, bias=True) for n in self.latent_names
        }
        if qk_rms_norm:
            self.sa_q_rms_norm: Dict[str, MultiHeadRMSNorm] = {
                n: MultiHeadRMSNorm(self.head_dim, num_heads) for n in self.latent_names
            }
            self.sa_k_rms_norm: Dict[str, MultiHeadRMSNorm] = {
                n: MultiHeadRMSNorm(self.head_dim, num_heads) for n in self.latent_names
            }
        if use_rope:
            self.sa_rope = RoPE(channels)

        # Cross-attn per-modality
        self.cross_attn: Dict[str, MultiHeadCrossAttention] = {
            n: MultiHeadCrossAttention(
                channels, ctx_channels, num_heads,
                qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
            )
            for n in self.latent_names
        }

        # MLP per-modality
        self.mlp: Dict[str, FeedForwardNet] = {
            n: FeedForwardNet(channels, mlp_ratio=mlp_ratio) for n in self.latent_names
        }

        self.share_mod = share_mod
        if not share_mod:
            assert cond_dim is not None
            self.adaLN_modulation = AdaLNModulation(cond_dim, channels)

    def _split_mod(self, mod: mx.array):
        c = mod.shape[-1] // 6
        return tuple(mod[:, i * c : (i + 1) * c] for i in range(6))

    def _self_attn(self, h: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Per-modality self-attention with protected-modality routing."""
        # 1. Project per modality
        q_d, k_d, v_d = {}, {}, {}
        shapes = {}
        for n, x in h.items():
            B, N, C = x.shape
            shapes[n] = (B, N, C)
            qkv = self.sa_to_qkv[n](x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            v = qkv[:, :, 2]
            if self.use_rope:
                q_t = q.transpose(0, 2, 1, 3)
                k_t = k.transpose(0, 2, 1, 3)
                q_t, k_t = self.sa_rope(q_t, k_t)
                q = q_t.transpose(0, 2, 1, 3)
                k = k_t.transpose(0, 2, 1, 3)
            if self.qk_rms_norm:
                q = self.sa_q_rms_norm[n](q)
                k = self.sa_k_rms_norm[n](k)
            q_d[n] = q  # (B, N, H, D)
            k_d[n] = k
            v_d[n] = v

        # 2. Routed attention
        out_d: Dict[str, mx.array] = {}
        # 2a. Protected modalities self-attend
        for n in self.protect_modality_list:
            if n not in q_d:
                continue
            B, N, C = shapes[n]
            q = q_d[n].transpose(0, 2, 1, 3)
            k = k_d[n].transpose(0, 2, 1, 3)
            v = v_d[n].transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out_d[n] = o.transpose(0, 2, 1, 3).reshape(B, N, C)

        # 2b. Other modalities attend to (others + protected) along token dim
        others = [n for n in self.latent_names if n not in self.protect_modality_list]
        if others:
            # concat others' q/k/v
            def _cat(d, names):
                return mx.concatenate([d[n] for n in names], axis=1)  # along N
            q_o = _cat(q_d, others)
            k_o = _cat(k_d, others)
            v_o = _cat(v_d, others)
            # protected k/v concat (no detach needed at inference)
            prot = [n for n in self.protect_modality_list if n in q_d]
            if prot:
                k_p = _cat(k_d, prot)
                v_p = _cat(v_d, prot)
                k_full = mx.concatenate([k_o, k_p], axis=1)
                v_full = mx.concatenate([v_o, v_p], axis=1)
            else:
                k_full = k_o
                v_full = v_o
            B = q_o.shape[0]
            # heads-first
            q_oh = q_o.transpose(0, 2, 1, 3)
            k_fh = k_full.transpose(0, 2, 1, 3)
            v_fh = v_full.transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(q_oh, k_fh, v_fh, scale=self.scale)
            o = o.transpose(0, 2, 1, 3)  # (B, N_o, H, D)
            # split back
            offset = 0
            for n in others:
                B_, N_, C_ = shapes[n]
                slice_ = o[:, offset : offset + N_]
                out_d[n] = slice_.reshape(B_, N_, C_)
                offset += N_

        # 3. Per-modality output projection
        return {n: self.sa_to_out[n](out_d[n]) for n in self.latent_names}

    def __call__(
        self,
        x: Dict[str, mx.array],
        mod: mx.array,
        context: mx.array,
    ) -> Dict[str, mx.array]:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_mod(mod)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod)

        # Self-attn branch
        h = {n: self.norm1[n](x[n]) for n in self.latent_names}
        h = {n: h[n] * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :] for n in self.latent_names}
        h = self._self_attn(h)
        h = {n: h[n] * gate_msa[:, None, :] for n in self.latent_names}
        x = {n: x[n] + h[n] for n in self.latent_names}

        # Cross-attn branch
        h = {n: self.norm2[n](x[n]) for n in self.latent_names}
        h = {n: self.cross_attn[n](h[n], context) for n in self.latent_names}
        x = {n: x[n] + h[n] for n in self.latent_names}

        # MLP branch
        h = {n: self.norm3[n](x[n]) for n in self.latent_names}
        h = {n: h[n] * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :] for n in self.latent_names}
        h = {n: self.mlp[n](h[n]) for n in self.latent_names}
        h = {n: h[n] * gate_mlp[:, None, :] for n in self.latent_names}
        x = {n: x[n] + h[n] for n in self.latent_names}

        return x


# ---------------------------------------------------------------------------
# Top-level backbones
# ---------------------------------------------------------------------------


class DiTBackbone(nn.Module):
    """Stack of single-modality cross-attn DiT blocks (matches slat_flow architecture).

    Forward signature::
        x       : (B, N, channels)
        cond    : (B, cond_channels)         # timestep + d embeds, summed
        context : (B, M, ctx_channels)       # image features
        indices : (B, N) or None             # for RoPE (positional)
    """

    def __init__(
        self,
        depth: int,
        channels: int,
        num_heads: int,
        ctx_channels: int,
        cond_channels: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        share_mod: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.num_heads = num_heads
        self.share_mod = share_mod
        # Default to fp32 internally; ``set_block_dtype`` flips per-block
        # weights to bf16 for the mixed-precision inference path.
        self.block_dtype = mx.float32

        self.t_embedder = TimestepEmbedder(channels)
        if share_mod:
            self.adaLN_modulation = AdaLNModulation(channels, channels)

        self.blocks = [
            DiTCrossBlock(
                channels=channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                cond_dim=channels,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
                share_mod=share_mod,
            )
            for _ in range(depth)
        ]

    def set_block_dtype(self, dtype) -> None:
        """Cast all per-block weights (Linear / RMSNorm / LayerNorm) to ``dtype``.

        t_embedder + (optional shared) adaLN_modulation stay fp32 so the
        sinusoidal timestep + mod cond are computed in fp32. Each block's
        own adaLN_modulation gets cast since the cond is bf16 by then.
        """
        self.block_dtype = dtype
        if dtype == mx.float32:
            return
        for blk in self.blocks:
            _cast_module_params_to_dtype(blk, dtype)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        indices: Optional[mx.array] = None,
    ) -> mx.array:
        # Time + AdaLN cond computed in fp32 (numerically sensitive).
        t_emb = self.t_embedder(t)
        if self.share_mod:
            mod = self.adaLN_modulation.proj(nn.silu(t_emb))
        else:
            mod = t_emb

        # bf16 mixed precision: cast block inputs (latent, mod, context) to
        # bf16 once at backbone entry; cast latent back to fp32 at exit.
        # See PT autocast(bfloat16) in inference_pipeline.py:71, 674.
        block_dtype = self.block_dtype
        if block_dtype != mx.float32:
            x = x.astype(block_dtype)
            mod = mod.astype(block_dtype)
            context_b = context.astype(block_dtype)
        else:
            context_b = context

        for blk in self.blocks:
            x = blk(x, mod, context_b, indices=indices)

        if block_dtype != mx.float32:
            x = x.astype(mx.float32)
        return x

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        prefix: str = "reverse_fn.backbone.",
        depth: int = 24,
        channels: int = 1024,
        num_heads: int = 16,
        ctx_channels: int = 1024,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        share_mod: bool = False,
        block_dtype=None,
    ) -> "DiTBackbone":
        sd = mx.load(npz_path)
        model = cls(
            depth=depth, channels=channels, num_heads=num_heads,
            ctx_channels=ctx_channels, cond_channels=channels,
            mlp_ratio=mlp_ratio, use_rope=use_rope,
            qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
            share_mod=share_mod,
        )

        def get(name):
            full = prefix + name
            if full not in sd:
                raise KeyError(full)
            return sd[full]

        # t_embedder.mlp.0 / .2
        model.t_embedder.fc0.weight = get("t_embedder.mlp.0.weight")
        model.t_embedder.fc0.bias = get("t_embedder.mlp.0.bias")
        model.t_embedder.fc2.weight = get("t_embedder.mlp.2.weight")
        model.t_embedder.fc2.bias = get("t_embedder.mlp.2.bias")

        for i in range(depth):
            blk = model.blocks[i]
            base = f"blocks.{i}"
            # adaLN
            if not share_mod:
                blk.adaLN_modulation.proj.weight = get(f"{base}.adaLN_modulation.1.weight")
                blk.adaLN_modulation.proj.bias = get(f"{base}.adaLN_modulation.1.bias")
            # norm2 (only affine norm)
            blk.norm2.weight = get(f"{base}.norm2.weight")
            blk.norm2.bias = get(f"{base}.norm2.bias")
            # self_attn
            blk.self_attn.to_qkv.weight = get(f"{base}.self_attn.to_qkv.weight")
            blk.self_attn.to_qkv.bias = get(f"{base}.self_attn.to_qkv.bias")
            blk.self_attn.to_out.weight = get(f"{base}.self_attn.to_out.weight")
            blk.self_attn.to_out.bias = get(f"{base}.self_attn.to_out.bias")
            if qk_rms_norm:
                blk.self_attn.q_rms_norm.gamma = get(f"{base}.self_attn.q_rms_norm.gamma")
                blk.self_attn.k_rms_norm.gamma = get(f"{base}.self_attn.k_rms_norm.gamma")
            # cross_attn
            blk.cross_attn.to_q.weight = get(f"{base}.cross_attn.to_q.weight")
            blk.cross_attn.to_q.bias = get(f"{base}.cross_attn.to_q.bias")
            blk.cross_attn.to_kv.weight = get(f"{base}.cross_attn.to_kv.weight")
            blk.cross_attn.to_kv.bias = get(f"{base}.cross_attn.to_kv.bias")
            blk.cross_attn.to_out.weight = get(f"{base}.cross_attn.to_out.weight")
            blk.cross_attn.to_out.bias = get(f"{base}.cross_attn.to_out.bias")
            if qk_rms_norm_cross:
                blk.cross_attn.q_rms_norm.gamma = get(f"{base}.cross_attn.q_rms_norm.gamma")
                blk.cross_attn.k_rms_norm.gamma = get(f"{base}.cross_attn.k_rms_norm.gamma")
            # mlp.mlp.0/2
            blk.mlp.fc0.weight = get(f"{base}.mlp.mlp.0.weight")
            blk.mlp.fc0.bias = get(f"{base}.mlp.mlp.0.bias")
            blk.mlp.fc2.weight = get(f"{base}.mlp.mlp.2.weight")
            blk.mlp.fc2.bias = get(f"{base}.mlp.mlp.2.bias")

        if block_dtype is not None and block_dtype != mx.float32:
            model.set_block_dtype(block_dtype)

        return model


class MOTDiTBackbone(nn.Module):
    """Multi-modality DiT backbone (matches ss_flow architecture).

    Forward signature::
        latents : Dict[str, (B, N_n, channels)]
        t       : (B,)                       # scalar timestep
        context : (B, M, ctx_channels)
        d       : optional shortcut step embedding (B,)
    """

    # Default modality config from observed npz keys.
    # NOTE: stage-1 ss_flow only routes two latent groups through the
    # transformer blocks (``shape`` and ``6drotation_normalized``). The PT
    # config's ``latent_share_transformer`` merges the lower-dim modalities
    # (scale / translation / translation_scale) into one of these two before
    # the backbone runs, then splits them back at output. Per-block weights
    # only carry "shape" and "6drotation_normalized" entries.
    DEFAULT_LATENT_NAMES = [
        "shape",
        "6drotation_normalized",
    ]

    def __init__(
        self,
        depth: int,
        channels: int,
        num_heads: int,
        ctx_channels: int,
        latent_names: List[str],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        share_mod: bool = False,
        has_d_embedder: bool = True,
        protect_modality_list: Optional[List[str]] = None,
    ):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.latent_names = list(latent_names)
        self.share_mod = share_mod
        self.has_d_embedder = has_d_embedder
        # Default fp32 internally; ``set_block_dtype`` flips to bf16.
        self.block_dtype = mx.float32

        self.t_embedder = TimestepEmbedder(channels)
        if has_d_embedder:
            self.d_embedder = TimestepEmbedder(channels)

        if share_mod:
            self.adaLN_modulation = AdaLNModulation(channels, channels)

        self.blocks = [
            MOTDiTCrossBlock(
                channels=channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                latent_names=latent_names,
                cond_dim=channels,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
                share_mod=share_mod,
                protect_modality_list=protect_modality_list,
            )
            for _ in range(depth)
        ]

    def __call__(
        self,
        latents: Dict[str, mx.array],
        t: mx.array,
        context: mx.array,
        d: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        # Time + (optional) shortcut + AdaLN cond computed in fp32.
        t_emb = self.t_embedder(t)
        if d is not None and self.has_d_embedder:
            t_emb = t_emb + self.d_embedder(d)
        mod = self.adaLN_modulation.proj(nn.silu(t_emb)) if self.share_mod else t_emb

        # bf16 mixed precision: cast each modality latent + mod + context
        # at backbone entry; cast outputs back to fp32 at exit.
        block_dtype = self.block_dtype
        h = dict(latents)
        if block_dtype != mx.float32:
            h = {k: v.astype(block_dtype) for k, v in h.items()}
            mod = mod.astype(block_dtype)
            context_b = context.astype(block_dtype)
        else:
            context_b = context

        for blk in self.blocks:
            h = blk(h, mod, context_b)

        if block_dtype != mx.float32:
            h = {k: v.astype(mx.float32) for k, v in h.items()}
        return h

    def set_block_dtype(self, dtype) -> None:
        """Cast all per-block weights to ``dtype`` (mirrors DiTBackbone)."""
        self.block_dtype = dtype
        if dtype == mx.float32:
            return
        for blk in self.blocks:
            _cast_module_params_to_dtype(blk, dtype)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        prefix: str = "reverse_fn.backbone.",
        depth: int = 24,
        channels: int = 1024,
        num_heads: int = 16,
        ctx_channels: int = 1024,
        latent_names: Optional[List[str]] = None,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        share_mod: bool = False,
        has_d_embedder: bool = True,
        protect_modality_list: Optional[List[str]] = None,
        block_dtype=None,
    ) -> "MOTDiTBackbone":
        sd = mx.load(npz_path)
        if latent_names is None:
            latent_names = cls.DEFAULT_LATENT_NAMES
        model = cls(
            depth=depth, channels=channels, num_heads=num_heads,
            ctx_channels=ctx_channels, latent_names=latent_names,
            mlp_ratio=mlp_ratio, use_rope=use_rope,
            qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
            share_mod=share_mod, has_d_embedder=has_d_embedder,
            protect_modality_list=protect_modality_list,
        )

        def get(name):
            full = prefix + name
            if full not in sd:
                raise KeyError(full)
            return sd[full]

        def maybe(name):
            full = prefix + name
            return sd.get(full)

        # t_embedder
        model.t_embedder.fc0.weight = get("t_embedder.mlp.0.weight")
        model.t_embedder.fc0.bias = get("t_embedder.mlp.0.bias")
        model.t_embedder.fc2.weight = get("t_embedder.mlp.2.weight")
        model.t_embedder.fc2.bias = get("t_embedder.mlp.2.bias")
        # d_embedder (optional)
        if has_d_embedder and maybe("d_embedder.mlp.0.weight") is not None:
            model.d_embedder.fc0.weight = get("d_embedder.mlp.0.weight")
            model.d_embedder.fc0.bias = get("d_embedder.mlp.0.bias")
            model.d_embedder.fc2.weight = get("d_embedder.mlp.2.weight")
            model.d_embedder.fc2.bias = get("d_embedder.mlp.2.bias")

        for i in range(depth):
            blk = model.blocks[i]
            base = f"blocks.{i}"
            if not share_mod:
                blk.adaLN_modulation.proj.weight = get(f"{base}.adaLN_modulation.1.weight")
                blk.adaLN_modulation.proj.bias = get(f"{base}.adaLN_modulation.1.bias")
            for n in latent_names:
                # norm2 (affine)
                blk.norm2[n].weight = get(f"{base}.norm2.{n}.weight")
                blk.norm2[n].bias = get(f"{base}.norm2.{n}.bias")
                # self_attn per-modality
                blk.sa_to_qkv[n].weight = get(f"{base}.self_attn.to_qkv.{n}.weight")
                blk.sa_to_qkv[n].bias = get(f"{base}.self_attn.to_qkv.{n}.bias")
                blk.sa_to_out[n].weight = get(f"{base}.self_attn.to_out.{n}.weight")
                blk.sa_to_out[n].bias = get(f"{base}.self_attn.to_out.{n}.bias")
                if qk_rms_norm:
                    blk.sa_q_rms_norm[n].gamma = get(f"{base}.self_attn.q_rms_norm.{n}.gamma")
                    blk.sa_k_rms_norm[n].gamma = get(f"{base}.self_attn.k_rms_norm.{n}.gamma")
                # cross_attn per-modality
                ca = blk.cross_attn[n]
                ca.to_q.weight = get(f"{base}.cross_attn.{n}.to_q.weight")
                ca.to_q.bias = get(f"{base}.cross_attn.{n}.to_q.bias")
                ca.to_kv.weight = get(f"{base}.cross_attn.{n}.to_kv.weight")
                ca.to_kv.bias = get(f"{base}.cross_attn.{n}.to_kv.bias")
                ca.to_out.weight = get(f"{base}.cross_attn.{n}.to_out.weight")
                ca.to_out.bias = get(f"{base}.cross_attn.{n}.to_out.bias")
                # mlp per-modality
                blk.mlp[n].fc0.weight = get(f"{base}.mlp.{n}.mlp.0.weight")
                blk.mlp[n].fc0.bias = get(f"{base}.mlp.{n}.mlp.0.bias")
                blk.mlp[n].fc2.weight = get(f"{base}.mlp.{n}.mlp.2.weight")
                blk.mlp[n].fc2.bias = get(f"{base}.mlp.{n}.mlp.2.bias")

        if block_dtype is not None and block_dtype != mx.float32:
            model.set_block_dtype(block_dtype)

        return model
