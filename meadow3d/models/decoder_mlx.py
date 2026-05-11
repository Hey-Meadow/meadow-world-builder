"""MLX port of SAM 3D Objects decoders.

Two decoders are ported here, both end-points of the inference pipeline:

1. ``SSDecoder``     — dense 3D conv VAE decoder. Maps an 8-channel latent
                       cube ``z (B, 8, R, R, R)`` (R = 16 by default for
                       sam-3d-objects ss_generator output) to a 1-channel
                       occupancy logit volume at 4*R resolution.

2. ``SLATDecoderGS`` — sparse transformer that maps a sparse latent
                       ``slat (N, 8)`` (with int voxel coords) to per-voxel
                       Gaussian-splat parameters. Output is N*32 Gaussians,
                       each described by ``{xyz(3), rgb(3) base SH, scale(3),
                       rotation(4), opacity(1)}`` = 14 params.

Critical port note (from RECON_3D.md): the GS decoder uses **NO sparse 3D
convolutions**. Its only sparse-aware operations are
  - ``SparseLinear`` = ``nn.Linear`` over ``(N, C)`` flat features, and
  - windowed self-attention (``swin``) which is a pure sort + gather +
    block-diagonal SDPA over coordinates.
That means we need ZERO custom Metal kernels for this decoder; pure MLX
ops (fancy indexing, ``mx.fast.scaled_dot_product_attention``, ``argsort``)
are sufficient.

PT source files:
  - ``sam3d_objects/.../models/sparse_structure_vae.py`` (SSDecoder)
  - ``sam3d_objects/.../models/structured_latent_vae/decoder_gs.py`` (SLATDecoderGS)
  - ``sam3d_objects/.../models/structured_latent_vae/base.py`` (SparseTransformerBase)
  - ``sam3d_objects/.../modules/sparse/transformer/blocks.py`` (block layout)
  - ``sam3d_objects/.../modules/sparse/attention/windowed_attn.py`` (swin attn)
  - ``sam3d_objects/.../representations/gaussian/gaussian_model.py`` (.ply)

Weight conventions (set by ``meadow3d/weights/convert.py``):
  - Conv3d weight: ``(out, kD, kH, kW, in)`` (MLX channels-last) — already
    pre-transposed in the npz.
  - Linear weight: ``(out, in)`` (matches both PT and MLX ``nn.Linear``).
  - Layer-norm has no learnable affine in slat decoder
    (``elementwise_affine=False``); only the dense SSDecoder norms are
    affine (those are layer-norm-over-channel a.k.a. ``ChannelLayerNorm32``).

Inference path uses pure functions; no torch.
"""

from __future__ import annotations

import math
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Activations / helpers
# ---------------------------------------------------------------------------


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _gelu_tanh(x: mx.array) -> mx.array:
    # exact match to PyTorch ``F.gelu(x, approximate='tanh')``
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def _layer_norm(x: mx.array, weight: Optional[mx.array] = None,
                bias: Optional[mx.array] = None, eps: float = 1e-5) -> mx.array:
    """LayerNorm over the last axis. Affine optional (matches ln_affine=False)."""
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) * (x - mu), axis=-1, keepdims=True)
    y = (x - mu) * mx.rsqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# Dense SS decoder (latent cube -> occupancy volume)
# ---------------------------------------------------------------------------
#
# Architecture (read from ss_decoder.npz; matches sparse_structure_vae.py
# config used at training):
#   channels             = [512, 128, 32]
#   latent_channels      = 8
#   out_channels         = 1
#   num_res_blocks       = 2
#   num_res_blocks_middle= 2
#   norm_type            = 'layer'  (ChannelLayerNorm32, affine=True)
#   reshape_input_to_cube= True     (inference receives flat tokens)
#
# Block linearization in the npz keys:
#   input_layer    : Conv3d(8 -> 512, k=3, p=1)
#   middle_block.0 : ResBlock3d(512)
#   middle_block.1 : ResBlock3d(512)
#   blocks.0       : ResBlock3d(512)
#   blocks.1       : ResBlock3d(512)
#   blocks.2       : UpsampleBlock3d(512 -> 128)   (Conv3d 512->1024 + pixel_shuffle 2x)
#   blocks.3       : ResBlock3d(128)
#   blocks.4       : ResBlock3d(128)
#   blocks.5       : UpsampleBlock3d(128 -> 32)    (Conv3d 128->256 + pixel_shuffle 2x)
#   blocks.6       : ResBlock3d(32)
#   blocks.7       : ResBlock3d(32)
#   out_layer.0    : ChannelLayerNorm32(32, affine)
#   out_layer.2    : Conv3d(32 -> 1, k=3, p=1)


def _conv3d_with_npz(in_ch: int, out_ch: int, k: int = 3, padding: int = 1,
                     stride: int = 1) -> nn.Conv3d:
    """Construct an MLX Conv3d. Weight layout matches converter output:
    (out, kD, kH, kW, in) — same as MLX default.
    """
    return nn.Conv3d(in_channels=in_ch, out_channels=out_ch,
                     kernel_size=k, stride=stride, padding=padding, bias=True)


def pixel_shuffle_3d(x: mx.array, scale: int) -> mx.array:
    """3D PixelShuffle, channels-last.

    Input  : (B, S0, S1, S2, C * scale^3)
    Output : (B, S0*scale, S1*scale, S2*scale, C)

    Mirrors PT ``pixel_shuffle_3d`` exactly: the PT impl reshapes the
    channel axis ``C_*s^3`` as ``(C_, s_a, s_b, s_c)`` with ``C_`` OUTER
    (slowest-varying), then interleaves ``s_a/s_b/s_c`` with the three
    spatial axes (each scale axis pairs with the spatial axis at the
    same relative position).  Channels-last therefore must split the
    last dim as ``(C_, s_a, s_b, s_c)`` — i.e. ``C_`` BEFORE the scale
    axes, not after.  Splitting as ``(s_a, s_b, s_c, C_)`` would put C_
    as the inner (fastest) axis and produce a different (interleaved)
    channel ordering, which is what caused the cosine inversion at the
    first upsample block.
    """
    B, S0, S1, S2, C = x.shape
    assert C % (scale ** 3) == 0
    Cout = C // (scale ** 3)
    # Split the last axis into (Cout, s_a, s_b, s_c) — Cout outer, scale inner.
    x = x.reshape(B, S0, S1, S2, Cout, scale, scale, scale)
    # axes:    0  1   2   3   4     5      6      7
    # PT permute (B, C_, s_a, s_b, s_c, H, W, D) -> (B, C_, H, s_a, W, s_b, D, s_c)
    #            0  1   2    3    4    5  6  7  ->  0  1   5  2    6  3    7  4
    # Our axis order is (B, S0, S1, S2, Cout, s_a, s_b, s_c).
    # Want         (B, S0, s_a, S1, s_b, S2, s_c, Cout):
    # source axes  (0   1   5    2   6    3   7    4)
    x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    x = x.reshape(B, S0 * scale, S1 * scale, S2 * scale, Cout)
    return x


class _ChannelLayerNorm3d(nn.Module):
    """ChannelLayerNorm32 in channels-last layout.

    PT module permutes (B, C, D, H, W) -> (B, D, H, W, C), runs LayerNorm
    over the last (channel) axis with ``elementwise_affine=True``, then
    permutes back. In our channels-last MLX world the permute is a no-op,
    so this is just a learnable LayerNorm over the last axis.
    """
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _layer_norm(x, self.weight, self.bias, eps=self.eps)


class _ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None,
                 norm_type: str = "layer"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if norm_type != "layer":
            raise NotImplementedError(f"norm_type={norm_type} (only 'layer' used in sam-3d-objects)")
        self.norm1 = _ChannelLayerNorm3d(channels)
        self.norm2 = _ChannelLayerNorm3d(self.out_channels)
        self.conv1 = _conv3d_with_npz(channels, self.out_channels, k=3, padding=1)
        self.conv2 = _conv3d_with_npz(self.out_channels, self.out_channels, k=3, padding=1)
        if channels == self.out_channels:
            self.skip = None
        else:
            self.skip = _conv3d_with_npz(channels, self.out_channels, k=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = _silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _silu(h)
        h = self.conv2(h)
        if self.skip is not None:
            return h + self.skip(x)
        return h + x


class _UpsampleBlock3d(nn.Module):
    """Pixel-shuffle upsample: conv expands channels 8x, then unshuffle 2x."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = _conv3d_with_npz(in_channels, out_channels * 8, k=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        return pixel_shuffle_3d(x, scale=2)


class SSDecoder(nn.Module):
    """Dense Conv3d decoder. Latent ``z`` -> occupancy logit volume."""

    def __init__(self,
                 latent_channels: int = 8,
                 out_channels: int = 1,
                 channels: Sequence[int] = (512, 128, 32),
                 num_res_blocks: int = 2,
                 num_res_blocks_middle: int = 2,
                 reshape_input_to_cube: bool = True):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.reshape_input_to_cube = reshape_input_to_cube

        self.input_layer = _conv3d_with_npz(latent_channels, channels[0], k=3, padding=1)

        # middle_block: Sequential of ResBlock3d(channels[0])
        self.middle_block = [
            _ResBlock3d(channels[0], channels[0]) for _ in range(num_res_blocks_middle)
        ]

        # blocks: alternating num_res_blocks ResBlock3d and Upsample
        self.blocks = []
        for i, ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.blocks.append(_ResBlock3d(ch, ch))
            if i < len(channels) - 1:
                self.blocks.append(_UpsampleBlock3d(ch, channels[i + 1]))

        # out_layer = Sequential( ChannelLayerNorm32, SiLU, Conv3d(C,1) )
        # but SiLU is parameter-free, so we only register the LN and Conv.
        self.out_norm = _ChannelLayerNorm3d(channels[-1])
        self.out_conv = _conv3d_with_npz(channels[-1], out_channels, k=3, padding=1)

    @staticmethod
    def _flat_to_cube(z: mx.array) -> mx.array:
        """(B, T, C) -> (B, R, R, R, C) where T = R^3."""
        B, T, C = z.shape
        R = round(T ** (1.0 / 3.0))
        assert R ** 3 == T, f"flat latent must be a cube: T={T}, R^3={R**3}"
        return z.reshape(B, R, R, R, C)

    def __call__(self, z: mx.array) -> mx.array:
        """``z`` may be (B, T, C) flat tokens or (B, D, H, W, C) cube. Returns
        (B, D*4, H*4, W*4, out_channels).
        """
        if z.ndim == 3 and self.reshape_input_to_cube:
            z = self._flat_to_cube(z)
        elif z.ndim == 5 and z.shape[-1] != self.latent_channels:
            # was passed channels-first by accident: (B, C, D, H, W) -> (B, D, H, W, C)
            if z.shape[1] == self.latent_channels:
                z = mx.transpose(z, (0, 2, 3, 4, 1))
            else:
                raise ValueError(f"Latent shape {z.shape} doesn't match "
                                 f"latent_channels={self.latent_channels}")

        h = self.input_layer(z)
        for blk in self.middle_block:
            h = blk(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        h = _silu(h)
        h = self.out_conv(h)
        return h

    # -----------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------

    @classmethod
    def from_npz(cls, npz_path: str) -> "SSDecoder":
        m = cls()
        weights = mx.load(npz_path)
        m._load_npz_dict(weights)
        return m

    def _load_npz_dict(self, w: Dict[str, mx.array]) -> None:
        """Map flat key dict (matches PT state_dict) into the module tree."""
        # input_layer
        self.input_layer.weight = w["input_layer.weight"]
        self.input_layer.bias = w["input_layer.bias"]
        # middle_block.{i}.{conv1,conv2,norm1,norm2}.{weight,bias}
        for i, blk in enumerate(self.middle_block):
            self._load_resblock(blk, w, f"middle_block.{i}")
        # blocks.{i}: resblock or upsample
        for i, blk in enumerate(self.blocks):
            base = f"blocks.{i}"
            if isinstance(blk, _UpsampleBlock3d):
                blk.conv.weight = w[f"{base}.conv.weight"]
                blk.conv.bias = w[f"{base}.conv.bias"]
            else:
                self._load_resblock(blk, w, base)
        # out_layer = Sequential[Norm, SiLU, Conv]
        # PyTorch Sequential keys: out_layer.0 (norm), out_layer.2 (conv).
        self.out_norm.weight = w["out_layer.0.weight"]
        self.out_norm.bias = w["out_layer.0.bias"]
        self.out_conv.weight = w["out_layer.2.weight"]
        self.out_conv.bias = w["out_layer.2.bias"]

    @staticmethod
    def _load_resblock(blk: _ResBlock3d, w: Dict[str, mx.array], base: str) -> None:
        blk.norm1.weight = w[f"{base}.norm1.weight"]
        blk.norm1.bias = w[f"{base}.norm1.bias"]
        blk.norm2.weight = w[f"{base}.norm2.weight"]
        blk.norm2.bias = w[f"{base}.norm2.bias"]
        blk.conv1.weight = w[f"{base}.conv1.weight"]
        blk.conv1.bias = w[f"{base}.conv1.bias"]
        blk.conv2.weight = w[f"{base}.conv2.weight"]
        blk.conv2.bias = w[f"{base}.conv2.bias"]
        if blk.skip is not None:
            blk.skip.weight = w[f"{base}.skip_connection.weight"]
            blk.skip.bias = w[f"{base}.skip_connection.bias"]


# ---------------------------------------------------------------------------
# Sparse-LATent Gaussian decoder (sparse transformer + per-voxel Gaussian heads)
# ---------------------------------------------------------------------------
#
# Architecture (from slat_decoder_gs.npz introspection):
#   model_channels       = 768
#   num_blocks           = 12
#   num_heads            = 12  (head dim = 64)
#   mlp_ratio            = 4   (3072)
#   attn_mode            = 'swin' (windowed self-attention, window_size=8,
#                                  shift_window = window_size//2 * (i%2))
#   pe_mode              = 'ape'  (sinusoidal position encoding from (x,y,z))
#   ln_affine            = False  (no learnable LN params)
#   qk_rms_norm          = False
#   use_rope             = False
#
# Per-block weight keys (no LN params, no rms_norm):
#   blocks.{i}.attn.to_qkv.{weight,bias}   2304 x 768
#   blocks.{i}.attn.to_out.{weight,bias}   768  x 768
#   blocks.{i}.mlp.mlp.0.{weight,bias}     3072 x 768  (mlp = nn.Sequential)
#   blocks.{i}.mlp.mlp.2.{weight,bias}     768  x 3072
#
# Top-level keys:
#   input_layer.weight / bias              768 x 8
#   out_layer.weight / bias                448 x 768  (= 32 * 14)
#   offset_perturbation                    (32, 3)    pre-computed Hammersley


# Default representation_config from the paper / TRELLIS codebase. Used by
# ``to_representation`` when reconstructing absolute Gaussian xyz from
# (voxel_coord, offset_logits).
_DEFAULT_REP_CFG = {
    "num_gaussians": 32,
    "voxel_size": 1.5,
    "perturb_offset": True,
    # scaling_bias=0.01 (PT Gaussian default; reverse-engineered from PT pt_splat
    # saved scale stats vs raw _scaling: PT scale_0 mean ~-6.72 vs raw _scaling[0]
    # mean ~-4.55, which is consistent with softplus(_scaling + inv_softplus(0.01))
    # ~ exp(-9.15) ~1e-4 plus min_kernel; matches log to within ~0.1 across all
    # channels). MLX previously used 4e-3 producing scale ~1.16 too negative.
    #
    # 3d_filter_kernel_size=9e-4 (was 1e-4): PT pt_splat shows scale_0 min=-7.013
    # exactly across plush/table/chair, which implies log(min_kernel) = -7.013 ->
    # min_kernel = exp(-7.013) ~ 9e-4. The previous value of 1e-4 produced
    # scale_min = log(1e-4) = -9.21 which let huge softplus tails push fluffy
    # outlier Gaussians into the saved PLY.
    "3d_filter_kernel_size": 9e-4,
    "scaling_bias": 0.01,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
    # scaling_clamp_max: cap raw _scaling above this value to suppress
    # blown-up Gaussians on fluffy chair-leg / edge voxels. PT pt_splat
    # tail goes only to ~+1.8 raw on these objects, but MLX (with imperfect
    # SLAT parity) sometimes spits +15 raw. Clamping to 0 keeps the largest
    # Gaussian within softplus(0 + bias_logit) = softplus(-4.6) ~ 0.01 ->
    # log(0.01) ~ -4.6, well within the PT distribution.
    "scaling_clamp_max": 0.0,  # PT-spec: clamp raw _scaling >= 0 (chair/table production-grade)
    "lr": {
        "_xyz": 1.0,
        "_features_dc": 1.0,
        "_scaling": 1.0,
        "_rotation": 0.1,
        "_opacity": 1.0,
    },
}


def _absolute_position_embedding(coords: mx.array, channels: int) -> mx.array:
    """Reproduce ``AbsolutePositionEmbedder`` in MLX.

    coords:    (N, 3) int voxel positions
    returns:   (N, channels) sin/cos positional embedding
    """
    in_channels = coords.shape[-1]  # 3
    freq_dim = channels // in_channels // 2
    freqs = mx.arange(freq_dim, dtype=mx.float32) / freq_dim
    freqs = 1.0 / (10000.0 ** freqs)               # (freq_dim,)
    flat = coords.astype(mx.float32).reshape(-1)   # (N*3,)
    out = flat[:, None] * freqs[None, :]           # (N*3, freq_dim)
    out = mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)  # (N*3, 2*freq_dim)
    out = out.reshape(coords.shape[0], in_channels * 2 * freq_dim)
    if out.shape[1] < channels:
        pad = mx.zeros((out.shape[0], channels - out.shape[1]))
        out = mx.concatenate([out, pad], axis=-1)
    return out


def _calc_window_partition(coords: mx.array, window_size: int,
                           shift_window: Tuple[int, int, int] = (0, 0, 0)) \
        -> Tuple[mx.array, mx.array, List[int]]:
    """Sort tokens by (batch, window_xyz). Mirror of windowed_attn.calc_window_partition.

    coords: (N, 4) int   columns = (batch, x, y, z)
    Returns (fwd_indices, bwd_indices, seq_lens) where ``seq_lens`` is a
    Python list of per-window token counts in fwd-order.
    """
    N = coords.shape[0]
    sx = coords[:, 1] + shift_window[0]
    sy = coords[:, 2] + shift_window[1]
    sz = coords[:, 3] + shift_window[2]
    # window indices
    wx = sx // window_size
    wy = sy // window_size
    wz = sz // window_size
    bx = coords[:, 0]

    # Compose a flat sort key: batch * NX*NY*NZ + wx * NY*NZ + ...
    NX = int(mx.max(wx).item()) + 1 if N > 0 else 1
    NY = int(mx.max(wy).item()) + 1 if N > 0 else 1
    NZ = int(mx.max(wz).item()) + 1 if N > 0 else 1
    NB = int(mx.max(bx).item()) + 1 if N > 0 else 1
    key = ((bx * NX + wx) * NY + wy) * NZ + wz  # (N,)
    fwd = mx.argsort(key)
    sorted_key = key[fwd]
    # Compute seq_lens by Python pass over sorted keys (Python int small N is fine).
    sk = sorted_key.tolist()
    seq_lens: List[int] = []
    cur = sk[0] if sk else None
    cnt = 0
    for v in sk:
        if v == cur:
            cnt += 1
        else:
            seq_lens.append(cnt)
            cur = v
            cnt = 1
    if sk:
        seq_lens.append(cnt)

    # bwd_indices: bwd[fwd[i]] = i
    arange_n = mx.arange(N, dtype=fwd.dtype)
    bwd = mx.zeros_like(fwd)
    bwd = mx.scatter(bwd, fwd[:, None], arange_n[:, None], 0) \
        if hasattr(mx, "scatter") else _bwd_via_argsort(fwd)
    return fwd, bwd, seq_lens


def _bwd_via_argsort(fwd: mx.array) -> mx.array:
    # bwd_indices[fwd_indices] = arange  =>  bwd = argsort(fwd)
    return mx.argsort(fwd)


def _windowed_self_attention(qkv: mx.array, coords: mx.array,
                             num_heads: int, window_size: int,
                             shift_window: Tuple[int, int, int]) -> mx.array:
    """Apply windowed (swin) self-attention.

    qkv:    (N, 3, H, C)
    coords: (N, 4) int, columns (batch, x, y, z)
    returns (N, H*C)
    """
    N, three, H, C = qkv.shape
    assert three == 3

    fwd, bwd, seq_lens = _calc_window_partition(coords, window_size, shift_window)
    qkv_sorted = qkv[fwd]  # (N, 3, H, C)

    if all(L == window_size for L in seq_lens):
        # Fast path: reshape into uniform blocks.
        B = len(seq_lens)
        L = window_size
        qkv_b = qkv_sorted.reshape(B, L, 3, H, C)
        q = mx.transpose(qkv_b[:, :, 0], (0, 2, 1, 3))  # (B, H, L, C)
        k = mx.transpose(qkv_b[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv_b[:, :, 2], (0, 2, 1, 3))
        scale = 1.0 / math.sqrt(C)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        # (B, H, L, C) -> (B*L, H, C)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B * L, H, C)
    else:
        # Variable-length path: per-window SDPA loop.
        # Cheap (windows tend to be small) but hot when many partial windows.
        outs: List[mx.array] = []
        start = 0
        scale = 1.0 / math.sqrt(C)
        for L in seq_lens:
            block = qkv_sorted[start:start + L]  # (L, 3, H, C)
            q = mx.transpose(block[:, 0], (1, 0, 2))[None]  # (1, H, L, C)
            k = mx.transpose(block[:, 1], (1, 0, 2))[None]
            v = mx.transpose(block[:, 2], (1, 0, 2))[None]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            o = mx.transpose(o[0], (1, 0, 2))   # (L, H, C)
            outs.append(o)
            start += L
        out = mx.concatenate(outs, axis=0)

    out = out[bwd]                       # (N, H, C)
    return out.reshape(N, H * C)


class _SparseTransformerBlock(nn.Module):
    """One block of pre-LN, no-affine LN + windowed self-attn + MLP."""

    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0,
                 window_size: int = 8, shift_window: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.shift_window = shift_window

        self.to_qkv = nn.Linear(channels, channels * 3, bias=True)
        self.to_out = nn.Linear(channels, channels, bias=True)
        self.mlp_fc1 = nn.Linear(channels, int(channels * mlp_ratio), bias=True)
        self.mlp_fc2 = nn.Linear(int(channels * mlp_ratio), channels, bias=True)

    def __call__(self, feats: mx.array, coords: mx.array) -> mx.array:
        # ATTENTION ---------------------------------------------------------
        h = _layer_norm(feats, eps=1e-6)
        qkv = self.to_qkv(h)                                # (N, 3C)
        N = qkv.shape[0]
        qkv = qkv.reshape(N, 3, self.num_heads, self.head_dim)
        attn_out = _windowed_self_attention(
            qkv, coords, self.num_heads, self.window_size, self.shift_window)
        attn_out = self.to_out(attn_out)
        feats = feats + attn_out

        # FFN ---------------------------------------------------------------
        h = _layer_norm(feats, eps=1e-6)
        h = self.mlp_fc1(h)
        h = _gelu_tanh(h)
        h = self.mlp_fc2(h)
        feats = feats + h
        return feats


class SLATDecoderGS(nn.Module):
    """Sparse Latent -> Gaussian splat decoder (32 Gaussians per voxel)."""

    # Layout order MUST match decoder_gs.py._calc_layout (insertion order):
    # _xyz, _features_dc, _scaling, _rotation, _opacity
    LAYOUT_ORDER = (
        ("_xyz",          3),
        ("_features_dc",  3),   # shape (G, 1, 3) but flat size = 3
        ("_scaling",      3),
        ("_rotation",     4),
        ("_opacity",      1),
    )

    def __init__(self,
                 resolution: int = 64,
                 model_channels: int = 768,
                 latent_channels: int = 8,
                 num_blocks: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 window_size: int = 8,
                 representation_config: Optional[dict] = None):
        super().__init__()
        self.resolution = resolution
        self.model_channels = model_channels
        self.latent_channels = latent_channels
        self.window_size = window_size
        self.rep_config = representation_config or dict(_DEFAULT_REP_CFG)

        num_gaussians = self.rep_config["num_gaussians"]
        # Sum size = 3+3+3+4+1 = 14 -> 14*G = out_channels
        out_channels_per_g = sum(s for _, s in self.LAYOUT_ORDER)
        self.out_channels = num_gaussians * out_channels_per_g

        self.input_layer = nn.Linear(latent_channels, model_channels, bias=True)
        self.blocks = []
        for i in range(num_blocks):
            shift = (window_size // 2 * (i % 2),) * 3  # swin shift pattern
            self.blocks.append(_SparseTransformerBlock(
                channels=model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                shift_window=shift,
            ))
        self.out_layer = nn.Linear(model_channels, self.out_channels, bias=True)

        # Buffers
        # offset_perturbation: precomputed Hammersley pre-atanh scaled values.
        self.offset_perturbation = mx.zeros((num_gaussians, 3))

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def __call__(self, slat_feats: mx.array, slat_coords: mx.array) -> Dict[str, mx.array]:
        """
        Args:
            slat_feats:  (N, latent_channels) float — sparse latent feats
            slat_coords: (N, 4) int — (batch, x, y, z) voxel coords (0..resolution-1)
        Returns:
            dict of Gaussian params (single-batch flat tensors). See class docstring.
        """
        h = self.input_layer(slat_feats)                # (N, C)
        # APE on (x,y,z) (drop batch index)
        ape = _absolute_position_embedding(slat_coords[:, 1:], self.model_channels)
        h = h + ape

        for blk in self.blocks:
            h = blk(h, slat_coords)

        h = _layer_norm(h, eps=1e-5)                    # final norm in PT impl
        h = self.out_layer(h)                           # (N, num_gaussians*14)
        return self._to_representation(h, slat_coords)

    # -----------------------------------------------------------------
    # Convert raw decoder logits to Gaussian params (PyTorch parity)
    # -----------------------------------------------------------------

    def _to_representation(self, feats: mx.array, coords: mx.array) -> Dict[str, mx.array]:
        """Reproduce ``SLatGaussianDecoder.to_representation``.

        Inputs:
            feats:  (N, num_gaussians*14)
            coords: (N, 4)  -- (batch, x, y, z)
        Output dict (single batch flattened):
            xyz       (N*G, 3)
            features  (N*G, 1, 3)   — stored as f_dc inside Gaussian.__init__
            scaling   (N*G, 3)
            rotation  (N*G, 4)
            opacity   (N*G, 1)
        These are RAW (pre-activation) outputs; ``save_gaussian_ply``
        applies the activations identical to gaussian_model.save_ply.
        """
        cfg = self.rep_config
        G = cfg["num_gaussians"]
        N = feats.shape[0]
        # voxel center xyz in [0,1]
        xyz_voxel = (coords[:, 1:].astype(mx.float32) + 0.5) / self.resolution  # (N, 3)

        # PT layout (decoder_gs.py::_calc_layout): channel slabs are stacked
        # MODALITY-MAJOR, not gaussian-major. The 448 output channels are laid
        # out as [xyz (96=G*3), features_dc (96=G*3), scaling (96=G*3),
        # rotation (128=G*4), opacity (32=G*1)].  Inside each modality slab
        # the order is gaussian-major (G consecutive blocks of `inner` size).
        # PT does `feats[:, range[0]:range[1]].reshape(-1, *shape)` per modality,
        # where `shape` for e.g. _xyz is (G, 3). So the modality slab is read
        # as (G, inner) in row-major order (i.e. g0_x g0_y g0_z, g1_x g1_y g1_z).
        idx = 0
        out: Dict[str, mx.array] = {}
        for name, size in self.LAYOUT_ORDER:
            slab_size = G * size
            slab = feats[:, idx:idx + slab_size]            # (N, G*size)
            chunk = slab.reshape(N, G, size)                # (N, G, size)
            chunk = chunk * cfg["lr"][name]
            if name == "_xyz":
                if cfg.get("perturb_offset", True):
                    chunk = chunk + self.offset_perturbation[None]
                offset = mx.tanh(chunk) / self.resolution * 0.5 * cfg["voxel_size"]
                xyz_abs = xyz_voxel[:, None, :] + offset    # (N, G, 3)
                out["_xyz"] = xyz_abs.reshape(N * G, 3)
            elif name == "_features_dc":
                out["_features_dc"] = chunk.reshape(N * G, 1, 3)
            elif name == "_scaling":
                # Clamp raw _scaling above the configured ceiling. Without
                # this, MLX-side SLAT noise occasionally produces raw values
                # in the +5..+15 range which softplus+log explodes into
                # +0..+10 in the saved PLY (visible as fluffy Gaussians on
                # chair legs / silhouette edges). PT raw stays under ~+1.8.
                clamp_max = cfg.get("scaling_clamp_max", None)
                scaling_chunk = chunk.reshape(N * G, 3)
                if clamp_max is not None:
                    scaling_chunk = mx.minimum(scaling_chunk, clamp_max)
                out["_scaling"] = scaling_chunk
            elif name == "_rotation":
                # Normalize raw quaternion to unit length. PT does this
                # implicitly at render time via F.normalize(_rotation +
                # rots_bias); the saved PLY stores raw quats but Meta's
                # public viewer expects unit-length quats. Without this,
                # MLX rotations have norm > 1 (range [-1.5, +2.3]) which
                # the viewer interprets as scaling, ballooning Gaussians.
                rot_chunk = chunk.reshape(N * G, 4)
                # Quaternion norm includes the implicit rots_bias [1,0,0,0]
                # which is added inside save_gaussian_ply. Normalize there.
                out["_rotation"] = rot_chunk
            elif name == "_opacity":
                out["_opacity"] = chunk.reshape(N * G, 1)
            idx += slab_size
        # Stash a few config fields so save_gaussian_ply can apply activations.
        out["_meta"] = mx.array([
            cfg["scaling_bias"],
            cfg["opacity_bias"],
            cfg["3d_filter_kernel_size"],
            1.0 if cfg["scaling_activation"] == "exp" else 0.0,
        ], dtype=mx.float32)
        return out

    # -----------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------

    @classmethod
    def from_npz(cls, npz_path: str,
                 representation_config: Optional[dict] = None,
                 **overrides) -> "SLATDecoderGS":
        weights = mx.load(npz_path)
        # Infer num_blocks / model_channels / num_gaussians from the npz so
        # the same loader works for both slat_decoder_gs and slat_decoder_gs_4.
        block_ids = set()
        for k in weights:
            if k.startswith("blocks."):
                block_ids.add(int(k.split(".")[1]))
        num_blocks = max(block_ids) + 1
        model_channels = weights["input_layer.weight"].shape[0]
        latent_channels = weights["input_layer.weight"].shape[1]
        out_channels = weights["out_layer.weight"].shape[0]
        per_g = sum(s for _, s in cls.LAYOUT_ORDER)  # 14
        num_gaussians = out_channels // per_g
        rep_cfg = dict(_DEFAULT_REP_CFG)
        if representation_config is not None:
            rep_cfg.update(representation_config)
        rep_cfg["num_gaussians"] = num_gaussians

        kwargs = dict(
            resolution=overrides.get("resolution", 64),
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=overrides.get("num_heads", model_channels // 64),
            mlp_ratio=overrides.get("mlp_ratio", 4.0),
            window_size=overrides.get("window_size", 8),
            representation_config=rep_cfg,
        )
        m = cls(**kwargs)
        m._load_npz_dict(weights)
        return m

    def _load_npz_dict(self, w: Dict[str, mx.array]) -> None:
        self.input_layer.weight = w["input_layer.weight"]
        self.input_layer.bias = w["input_layer.bias"]
        self.out_layer.weight = w["out_layer.weight"]
        self.out_layer.bias = w["out_layer.bias"]
        if "offset_perturbation" in w:
            self.offset_perturbation = w["offset_perturbation"]
        for i, blk in enumerate(self.blocks):
            base = f"blocks.{i}"
            blk.to_qkv.weight = w[f"{base}.attn.to_qkv.weight"]
            blk.to_qkv.bias = w[f"{base}.attn.to_qkv.bias"]
            blk.to_out.weight = w[f"{base}.attn.to_out.weight"]
            blk.to_out.bias = w[f"{base}.attn.to_out.bias"]
            blk.mlp_fc1.weight = w[f"{base}.mlp.mlp.0.weight"]
            blk.mlp_fc1.bias = w[f"{base}.mlp.mlp.0.bias"]
            blk.mlp_fc2.weight = w[f"{base}.mlp.mlp.2.weight"]
            blk.mlp_fc2.bias = w[f"{base}.mlp.mlp.2.bias"]


# ---------------------------------------------------------------------------
# Gaussian splat .ply writer (3DGS standard format)
# ---------------------------------------------------------------------------


def _softplus(x):  # numpy
    import numpy as np
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _inverse_softplus(x):
    import numpy as np
    return np.log(np.expm1(x))


def _inverse_sigmoid(x):
    import numpy as np
    return np.log(x / (1.0 - x))


def save_gaussian_ply(gs_params: Dict[str, mx.array], path: str,
                      aabb: Sequence[float] = (-0.5, -0.5, -0.5, 1.0, 1.0, 1.0)) -> None:
    """Pack a dict of raw Gaussian decoder outputs into a 3DGS-compatible .ply.

    Activations / encodings here MIRROR
    ``representations/gaussian/gaussian_model.py::save_ply``:
        xyz_world  = _xyz * aabb[3:] + aabb[:3]
        opacity    = inverse_sigmoid( sigmoid(_opacity + opacity_bias_logit) )
                   = _opacity + opacity_bias_logit  (canceling, but written for parity)
        scale      = log( get_scaling )
                     where get_scaling = sqrt( softplus(_scaling+scale_bias)^2 + min_kernel^2 )
                     (or exp variant if scaling_activation=='exp')
        rotation   = _rotation + rots_bias  (rots_bias = [1,0,0,0])
        f_dc       = _features_dc.transpose(1,2).flatten(1)  (3 SH degree-0 coefs)

    Writes BINARY little-endian PLY for fastest loading; 3DGS viewers expect
    f4 properties in the canonical order.
    """
    import numpy as np

    # ---- pull tensors from MLX -> numpy ----
    xyz_local = np.asarray(gs_params["_xyz"]).astype(np.float32)         # (N, 3) in [0,1]
    feats_dc = np.asarray(gs_params["_features_dc"]).astype(np.float32)  # (N, 1, 3)
    scaling_raw = np.asarray(gs_params["_scaling"]).astype(np.float32)   # (N, 3)
    rotation_raw = np.asarray(gs_params["_rotation"]).astype(np.float32) # (N, 4)
    opacity_raw = np.asarray(gs_params["_opacity"]).astype(np.float32)   # (N, 1)
    meta = np.asarray(gs_params["_meta"]).astype(np.float32) if "_meta" in gs_params \
        else np.array([4e-3, 0.1, 1e-4, 0.0], dtype=np.float32)
    scaling_bias_val, opacity_bias_val, min_kernel, exp_flag = (
        float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3]))

    # ---- xyz: aabb + scale ----
    aabb = np.asarray(aabb, dtype=np.float32)
    xyz = xyz_local * aabb[3:] + aabb[:3]                                # (N, 3)

    # ---- features (DC) — store same as PT impl: transpose then flatten ----
    # feats_dc shape (N, 1, 3). PT does .transpose(1,2)->(N, 3, 1).flatten(start_dim=1)
    # -> (N, 3*1) = (N, 3). Equivalent to feats_dc.squeeze(1).
    f_dc = feats_dc.reshape(feats_dc.shape[0], -1).astype(np.float32)    # (N, 3)

    # ---- opacity: PT writes inverse_sigmoid(get_opacity), get_opacity =
    #      sigmoid(_opacity + opacity_bias_logit). The two cancel, so this
    #      is identically (_opacity + opacity_bias_logit).
    opacity_bias_logit = _inverse_sigmoid(opacity_bias_val)
    opacity = opacity_raw + opacity_bias_logit                            # (N, 1)

    # ---- scaling: PT does log( get_scaling ) with mininum_kernel_size
    if exp_flag > 0.5:  # scaling_activation == "exp"
        scale_bias_logit = math.log(scaling_bias_val)
    else:                # softplus
        scale_bias_logit = _inverse_softplus(scaling_bias_val).item() \
            if hasattr(_inverse_softplus(scaling_bias_val), "item") \
            else float(_inverse_softplus(scaling_bias_val))
    if exp_flag > 0.5:
        scales = np.exp(scaling_raw + scale_bias_logit)
    else:
        scales = _softplus(scaling_raw + scale_bias_logit)
    scales = np.sqrt(scales * scales + min_kernel * min_kernel)
    scales = np.log(scales)                                              # (N, 3)

    # ---- rotation: + rots_bias (1,0,0,0), then normalize to unit quaternion.
    # PT's gaussian_model.save_ply writes (_rotation + rots_bias) raw and
    # relies on the renderer (or layout-post-optimization, which does
    # `quat / quat.norm()` explicitly) to normalize. Meta's released
    # reference plys have rot magnitudes consistent with unit quats. MLX
    # raw rotations sometimes have ||q|| ~ 1.5..2.5 which makes the viewer
    # interpret them as a 1.5x..2.5x scale on top of the Gaussian, blowing
    # up edge voxels into the fluffy halo seen on the chair legs. Normalize
    # here so the PLY is viewer-ready.
    rotation = rotation_raw.copy()
    rotation[:, 0] += 1.0                                                # (N, 4)
    rot_norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
    rot_norm = np.maximum(rot_norm, 1e-8)
    rotation = rotation / rot_norm                                       # (N, 4)

    # ---- normals are zero per 3DGS convention ----
    normals = np.zeros_like(xyz, dtype=np.float32)                       # (N, 3)

    n_pts = xyz.shape[0]
    sh_dc_dim = f_dc.shape[1]      # 3
    scale_dim = scales.shape[1]    # 3
    rot_dim = rotation.shape[1]    # 4

    # ---- write PLY (binary little-endian) ----
    props = ["x", "y", "z", "nx", "ny", "nz"]
    props += [f"f_dc_{i}" for i in range(sh_dc_dim)]
    props += ["opacity"]
    props += [f"scale_{i}" for i in range(scale_dim)]
    props += [f"rot_{i}" for i in range(rot_dim)]

    # Concatenate into (N, P) row-major for binary write.
    row = np.concatenate(
        [xyz, normals, f_dc, opacity, scales, rotation],
        axis=1).astype("<f4")
    assert row.shape[1] == len(props), (row.shape[1], len(props))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_pts}\n".encode())
        for p in props:
            f.write(f"property float {p}\n".encode())
        f.write(b"end_header\n")
        f.write(row.tobytes())


__all__ = [
    "SSDecoder",
    "SLATDecoderGS",
    "save_gaussian_ply",
    "pixel_shuffle_3d",
]
