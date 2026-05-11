"""MLX port of SAM 3D Objects conditioning embedders.

Source modules (PyTorch, read-only):
  sam3d_objects/model/backbone/dit/embedder/dino.py            -> Dino (timm-style ViT trunk via torch.hub)
  sam3d_objects/model/backbone/dit/embedder/pointmap.py        -> PointPatchEmbed
  sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py  -> EmbedderFuser
  sam3d_objects/model/backbone/tdfy_dit/models/timestep_embedder.py -> TimestepEmbedder

Weight inspection (`meadow3d/weights/sam3d_objects/ss_embedder.npz`):
  - module_list.0.backbone.* : DINOv2 ViT-L/14 + register tokens
        embed_dim=1024, depth=24, num_heads=16 (head_dim=64), patch_size=14,
        num_register_tokens=4, pos_embed=(1,1370,1024)  -> 1 cls + 37*37 patches
        Has LayerScale (ls1.gamma, ls2.gamma) per block.
  - module_list.1.backbone.* : Same DINOv2 ViT-L/14 (separate weights, e.g. for masked/2nd image)
  - module_list.2.* : PointPatchEmbed
        embed_dim=512, patch_size=8, input_size=256, single transformer block (depth=1),
        num_heads=16 (head_dim=32), pos_embed=(1, 512, 32, 32), pos_embed_window=(1, 65, 512),
        cls_token=(1,1,512), invalid_xyz_token=(512,), point_proj=Linear(3, 512).
  - idx_emb : (3, 1024) learned per-modality positional embedding (added to projected tokens).
  - projection_nets.{i}.0 : LayerNorm(input_dim) (weight, bias)
    projection_nets.{i}.1 : FeedForward(w1, w2, w3) [llama3 SwiGLU FFN]
        i=0,1: dim=1024, hidden=2816, output=1024
        i=2  : dim=512,  hidden=2816, output=1024

slat_embedder.npz : same structure with only modules 0 and 1 (no point branch).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal + 2-layer MLP).
# Mirrors `tdfy_dit/models/timestep_embedder.TimestepEmbedder`.
# Lives in the flow npz (`{ss,slat}_flow.npz`), not the embedder npz, but we
# expose it here per SPEC_EMBED.md for self-contained imports.
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding -> SiLU -> Linear -> SiLU? No: PT uses

        Linear(freq_size, hidden) -> SiLU -> Linear(hidden, hidden)

    so the second linear is the output and there is no trailing SiLU.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        # PT uses nn.Sequential([Linear, SiLU, Linear]); to keep loading simple
        # we expose `mlp_0` and `mlp_2` and apply silu in __call__.
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
        if t.ndim == 0:
            t = t[None]
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        freq = self.timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        h = self.mlp_0(freq)
        h = nn.silu(h)
        h = self.mlp_2(h)
        return h


# ---------------------------------------------------------------------------
# DINOv2 ViT-L/14 with register tokens (timm-style "vit_large_patch14_dinov2").
# Required because PT loads via torch.hub; we replicate the architecture here.
# ---------------------------------------------------------------------------


class _LayerScale(nn.Module):
    """Learnable per-channel scale (gamma)."""

    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        self.gamma = mx.full((dim,), init_value, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class _Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class _DinoBlock(nn.Module):
    """Pre-norm transformer block with LayerScale (ls1, ls2)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = _Attention(dim=dim, num_heads=num_heads, qkv_bias=True)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _Mlp(dim, mlp_ratio=mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _DinoPatchEmbed(nn.Module):
    """Conv2d patch embed for DINO. PT (out, in, k, k) was already converted
    to (out, k, k, in) by meadow3d/weights/convert.py. MLX nn.Conv2d expects
    weight shape (out, kH, kW, in_c) and channels-last input, so we use it
    directly.
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        # x: (B, H, W, C) NHWC
        x = self.proj(x)  # (B, Hp, Wp, embed_dim)
        B, Hp, Wp, C = x.shape
        return x.reshape(B, Hp * Wp, C), (Hp, Wp)


class DinoViT(nn.Module):
    """DINOv2 ViT-L/14 with register tokens.

    Mirrors the tensor layout of the timm `vit_large_patch14_reg4_dinov2`
    family. Operates in channels-last (NHWC) MLX convention.

    Forward returns the concatenated [cls, patch_tokens] sequence (PT's
    `_forward_last_layer` with `prenorm_features=False`):

        out = concat([norm(cls).unsqueeze(1), norm(patch_tokens)], dim=1)

    shape (B, 1 + Hp*Wp, embed_dim). Register tokens are consumed inside the
    transformer but not returned (matches PT API).
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        norm_eps: float = 1e-6,
        normalize_images: bool = True,
        keep_register_tokens: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_register_tokens = num_register_tokens
        self.normalize_images = normalize_images
        self.keep_register_tokens = keep_register_tokens

        self.patch_embed = _DinoPatchEmbed(
            patch_size=patch_size, in_channels=3, embed_dim=embed_dim
        )

        # Tokens (named to match PT keys). Initialised to zeros; weights override.
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.mask_token = mx.zeros((1, embed_dim))  # unused at inference, kept for loading
        self.register_tokens = mx.zeros((1, num_register_tokens, embed_dim))

        # PT pos_embed shape (1, 1+num_patches, embed_dim) for canonical img_size.
        # 518/14 = 37 -> 37*37 = 1369 patches -> pos_embed length 1370.
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = mx.zeros((1, n_patches + 1, embed_dim))

        self.blocks = [
            _DinoBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, eps=norm_eps)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # ImageNet mean/std for the RGB preprocessing path (matches PT Dino).
        self._imagenet_mean = mx.array([0.485, 0.456, 0.406], dtype=mx.float32).reshape(1, 1, 1, 3)
        self._imagenet_std = mx.array([0.229, 0.224, 0.225], dtype=mx.float32).reshape(1, 1, 1, 3)

    # ---- helpers --------------------------------------------------------

    def _interpolate_pos_embed(self, Hp: int, Wp: int) -> mx.array:
        """Bilinearly interpolate pos_embed (cls held out) to the grid (Hp, Wp).

        Falls back to `pos_embed` directly when the canonical grid matches.
        """
        n_patches = Hp * Wp
        if self.pos_embed.shape[1] == n_patches + 1:
            return self.pos_embed
        # split cls / patches
        cls_pe = self.pos_embed[:, :1]
        patch_pe = self.pos_embed[:, 1:]
        L = patch_pe.shape[1]
        old = int(round(math.sqrt(L)))
        # reshape to (1, old, old, C)
        patch_pe = patch_pe.reshape(1, old, old, self.embed_dim)
        # MLX: simple bilinear via mx.image-like is not built-in; do nearest
        # via repeat then crop only when sizes match an integer ratio. For the
        # common case (canonical match), we never hit this branch.
        # General fallback: do linear interp manually along H then W.
        patch_pe = _bilinear_resize_2d(patch_pe, Hp, Wp)
        patch_pe = patch_pe.reshape(1, Hp * Wp, self.embed_dim)
        return mx.concatenate([cls_pe, patch_pe], axis=1)

    def _preprocess(self, image: mx.array) -> mx.array:
        """Normalize and (if needed) resize. Input NHWC float.

        If the caller provides an already square image whose side is a multiple
        of ``patch_size`` (e.g. 1024 for DINOv2/14 high-res inference), we skip
        the resize entirely and let the patch_embed produce a (Hp, Wp) grid
        whose pos_embed is bilinearly interpolated from the canonical 37x37.
        Otherwise we fall back to resizing to ``self.img_size`` (PT default).
        """
        B, H, W, C = image.shape
        ps = self.patch_size
        is_valid_grid = (H == W) and (H % ps == 0) and (H >= ps)
        target = H if is_valid_grid else self.img_size
        if (H, W) != (target, target):
            image = _bilinear_resize_2d(image, target, target)
        if C == 1:
            image = mx.broadcast_to(image, (B, target, target, 3))
        if self.normalize_images:
            image = (image - self._imagenet_mean) / self._imagenet_std
        return image

    # ---- forward --------------------------------------------------------

    def __call__(self, image: mx.array) -> mx.array:
        """image: (B, H, W, 3) NHWC float in [0, 1]. Returns (B, 1+N, C)."""
        x = self._preprocess(image)  # (B, S, S, 3)
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)  # (B, N, C)
        # add cls + register; pos_embed is added to [cls, patches] only (DINOv2 reg variant).
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x_cls = mx.concatenate([cls, x], axis=1)  # (B, 1+N, C)
        pe = self._interpolate_pos_embed(Hp, Wp)
        x_cls = x_cls + pe
        # interleave register tokens after cls (PT order: cls, reg, patches)
        if self.num_register_tokens > 0:
            reg = mx.broadcast_to(self.register_tokens, (B, self.num_register_tokens, self.embed_dim))
            x_full = mx.concatenate([x_cls[:, :1], reg, x_cls[:, 1:]], axis=1)
        else:
            x_full = x_cls
        for blk in self.blocks:
            x_full = blk(x_full)
        # PT chunk sizes (verified from dumped ss_cond / slat_cond):
        #   ss_cond  DINO chunk = 1370 (= 1+1369; register tokens DROPPED)
        #   slat_cond DINO chunk = 1374 (= 1+4+1369; register tokens KEPT)
        # The asymmetry is driven by `prenorm_features` flag in PT's `dino.py`
        # (read-only: sam3d_objects/model/backbone/dit/embedder/dino.py:95-108):
        # - prenorm_features=False -> cat([x_norm_clstoken, x_norm_patchtokens])
        #   i.e. the trained `self.norm` is applied THEN registers are dropped.
        # - prenorm_features=True  -> F.layer_norm(x_prenorm, [-1])
        #   i.e. UNWEIGHTED LN (gamma=1, beta=0) applied to the FULL prenorm features
        #   (cls+register+patches; trained `self.norm` is BYPASSED).
        # We pick the branch based on `keep_register_tokens` (set at load time
        # via npz filename: slat_embedder.npz -> True).
        if getattr(self, "keep_register_tokens", False):
            # prenorm_features=True: unweighted LN over feature dim, keep all tokens.
            mean = mx.mean(x_full, axis=-1, keepdims=True)
            var = mx.var(x_full, axis=-1, keepdims=True)
            x_norm = (x_full - mean) / mx.sqrt(var + 1e-5)
            return x_norm
        # prenorm_features=False: trained final norm, drop registers.
        x_full = self.norm(x_full)
        cls_out = x_full[:, :1]
        patch_out = x_full[:, 1 + self.num_register_tokens :]
        return mx.concatenate([cls_out, patch_out], axis=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bilinear_resize_2d(x: mx.array, out_h: int, out_w: int) -> mx.array:
    """Bilinear resize for (B, H, W, C). Pure-MLX implementation.

    Used by DINO pos_embed interpolation and image preprocessing. The hot path
    (canonical 518x518) skips this entirely.
    """
    B, H, W, C = x.shape
    if H == out_h and W == out_w:
        return x
    # Compute sampling grid (align_corners=False, half-pixel centers).
    def _coords(in_n: int, out_n: int) -> Tuple[mx.array, mx.array, mx.array]:
        # PyTorch align_corners=False: x = (i+0.5) * in/out - 0.5
        i = mx.arange(out_n, dtype=mx.float32)
        src = (i + 0.5) * (in_n / out_n) - 0.5
        src = mx.clip(src, 0.0, in_n - 1.0)
        i0 = mx.floor(src).astype(mx.int32)
        i1 = mx.minimum(i0 + 1, in_n - 1)
        w1 = src - i0.astype(mx.float32)
        w0 = 1.0 - w1
        return i0, i1, w0, w1

    h0, h1, wh0, wh1 = _coords(H, out_h)
    w0, w1, ww0, ww1 = _coords(W, out_w)

    # gather rows then cols
    x_h0 = x[:, h0, :, :]  # (B, out_h, W, C)
    x_h1 = x[:, h1, :, :]
    x_h = x_h0 * wh0[None, :, None, None] + x_h1 * wh1[None, :, None, None]
    x_w0 = x_h[:, :, w0, :]  # (B, out_h, out_w, C)
    x_w1 = x_h[:, :, w1, :]
    out = x_w0 * ww0[None, None, :, None] + x_w1 * ww1[None, None, :, None]
    return out


# ---------------------------------------------------------------------------
# Image embedder (wraps DinoViT; matches PT `Dino` API surface).
# ---------------------------------------------------------------------------


class ImageEmbedder(nn.Module):
    """DINO ViT trunk -> tokens. PT key prefix: `module_list.{i}.backbone.*`."""

    def __init__(
        self,
        input_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        normalize_images: bool = True,
        keep_register_tokens: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.backbone = DinoViT(
            img_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_register_tokens=num_register_tokens,
            normalize_images=normalize_images,
            keep_register_tokens=keep_register_tokens,
        )

    def __call__(self, image: mx.array) -> mx.array:
        return self.backbone(image)


# ---------------------------------------------------------------------------
# Point patch embedder (PT `PointPatchEmbed`).
# ---------------------------------------------------------------------------


class _PointBlock(nn.Module):
    """timm-style Block with `mlp_ratio=2.0` used inside PointPatchEmbed."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = _Attention(dim=dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _Mlp(dim, mlp_ratio=mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PointPatchEmbedder(nn.Module):
    """Encodes a pointmap (B, 3, H, W) into per-window tokens (B, N_win, D).

    Pure-MLX port of PT `PointPatchEmbed.forward`. PT yaml for SAM 3D Objects
    uses ``remap_output='linear'`` (no remap) for the SS pointmap branch, so
    that is the default here.

    All math operates in NHWC.
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 512,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        depth: int = 1,
        remap_output: str = "linear",
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.remap_output = remap_output

        self.point_proj = nn.Linear(3, embed_dim)
        self.invalid_xyz_token = mx.zeros((embed_dim,))
        num_patches = input_size // patch_size  # 32 for 256/8
        self.pos_embed = mx.zeros((1, embed_dim, num_patches, num_patches))
        self.pos_embed_window = mx.zeros((1, 1 + patch_size * patch_size, embed_dim))
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.blocks = [
            _PointBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ]

    # ---- helpers --------------------------------------------------------

    def _remap(self, points: mx.array) -> mx.array:
        """PT PointRemapper.exp: xy = xy_scaled / (1 + z_exp); z = log1p(z_exp)."""
        if self.remap_output == "linear":
            return points
        if self.remap_output == "exp":
            xy = points[..., :2]
            z_exp = points[..., 2:3]
            z = mx.log1p(z_exp)
            xy = xy / (1.0 + z_exp)
            return mx.concatenate([xy, z], axis=-1)
        # Other modes (sinh, exp_disparity, sinh_exp) unused by SAM 3D Objects.
        raise NotImplementedError(f"remap_output={self.remap_output!r} not supported in MLX port")

    def _resize_pointmap(self, xyz: mx.array) -> mx.array:
        """xyz: (B, 3, H, W) -> NHWC and resize to (input_size, input_size).

        PT uses `mode='nearest'`. We implement nearest via index gather.
        """
        B, C, H, W = xyz.shape
        # transpose to NHWC
        xyz = xyz.transpose(0, 2, 3, 1)  # (B, H, W, 3)
        if (H, W) != (self.input_size, self.input_size):
            # nearest neighbour
            yi = mx.minimum(
                ((mx.arange(self.input_size, dtype=mx.float32) + 0.5) * (H / self.input_size)).astype(mx.int32),
                H - 1,
            )
            xi = mx.minimum(
                ((mx.arange(self.input_size, dtype=mx.float32) + 0.5) * (W / self.input_size)).astype(mx.int32),
                W - 1,
            )
            xyz = xyz[:, yi, :, :][:, :, xi, :]
        return xyz

    def _get_pos_embed(self, h: int, w: int) -> mx.array:
        """Bilinearly resize self.pos_embed (1, C, Hp, Wp) -> (1, h, w, C)."""
        # convert to NHWC
        pe = self.pos_embed.transpose(0, 2, 3, 1)  # (1, Hp, Wp, C)
        pe = _bilinear_resize_2d(pe, h, w)  # (1, h, w, C)
        return pe

    # ---- forward --------------------------------------------------------

    def __call__(self, xyz: mx.array, valid_mask: Optional[mx.array] = None) -> mx.array:
        """xyz: (B, 3, H, W) -> tokens (B, N_win, D)."""
        x = self._resize_pointmap(xyz)  # (B, S, S, 3)
        B, H, W, _ = x.shape
        if valid_mask is None:
            # NaN/Inf mask (PT uses xyz.isfinite().all(dim=-1))
            valid_mask = mx.all(mx.isfinite(x), axis=-1)  # (B, H, W)
        # Replace invalid points with 0 before remap (avoid NaN propagation).
        x_safe = mx.where(valid_mask[..., None], x, mx.zeros_like(x))
        x_remapped = self._remap(x_safe)
        h = self.point_proj(x_remapped)  # (B, H, W, D)
        # Stop gradient + add invalid token to invalid positions.
        invalid = (~valid_mask)[..., None]  # (B, H, W, 1)
        h = mx.where(invalid, mx.broadcast_to(self.invalid_xyz_token, h.shape), h)

        # ---- inner_forward ----
        ps = self.patch_size
        nh = H // ps
        nw = W // ps
        # reshape to (B, nh, ps, nw, ps, D) -> (B, nh, nw, ps, ps, D)
        h = h.reshape(B, nh, ps, nw, ps, self.embed_dim)
        h = h.transpose(0, 1, 3, 2, 4, 5)
        h = h.reshape(B * nh * nw, ps * ps, self.embed_dim)

        cls = mx.broadcast_to(self.cls_token, (h.shape[0], 1, self.embed_dim))
        toks = mx.concatenate([cls, h], axis=1)  # (B*nh*nw, 1+ps*ps, D)
        toks = toks + self.pos_embed_window  # (1, 1+ps*ps, D)
        for blk in self.blocks:
            toks = blk(toks)
        # extract CLS, reshape to (B, nh*nw, D)
        win_emb = toks[:, 0].reshape(B, nh * nw, self.embed_dim)
        # add positional embedding for the window grid (PT: bilinear-resize from canonical)
        pos_patch = self._get_pos_embed(nh, nw).reshape(1, nh * nw, self.embed_dim)
        return win_emb + pos_patch


# ---------------------------------------------------------------------------
# Embedder fuser (PT `EmbedderFuser` with use_pos_embedding="learned", proj=on).
# ---------------------------------------------------------------------------


class _LlamaFFN(nn.Module):
    """Llama3 SwiGLU FFN used by EmbedderFuser projection.

    Layout (matches PT):
        w1 : Linear(dim, hidden, bias=False)
        w3 : Linear(dim, hidden, bias=False)
        w2 : Linear(hidden, output_dim, bias=False)

        out = w2(silu(w1(x)) * w3(x))
    """

    def __init__(self, dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, output_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


def _make_projection_net(input_dim: int, output_dim: int, multiplier: float = 4.0,
                         multiple_of: int = 256, ffn_dim_multiplier: Optional[float] = None
                         ) -> Tuple[nn.LayerNorm, _LlamaFFN]:
    """Replicate `EmbedderFuser._make_projection_net` (with projection_pre_norm=True)."""
    pre_norm = nn.LayerNorm(input_dim)
    # Llama3 hidden dim recipe (FeedForward.__init__):
    raw_hidden = int(multiplier * output_dim)
    hidden = int(2 * raw_hidden / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return pre_norm, _LlamaFFN(input_dim, hidden, output_dim)


class EmbedderFuser(nn.Module):
    """Concatenate per-modality tokens, with per-modality projection + learned
    positional bias (one row per modality kwarg).

    Args:
        embedders: list of nn.Module condition embedders (image/point).
        kwargs_info: per-embedder list of (kwarg_name, pos_group). PT allows
            multiple kwargs per embedder; we keep that flexibility.
        embed_dims: target output embed dim (max over embedder.embed_dim).
        n_pos_groups: number of distinct positional groups across all kwargs.
        projection_multiplier: hidden_dim multiplier for projection FFN.
    """

    def __init__(
        self,
        embedders: List[nn.Module],
        kwargs_info: List[List[Tuple[str, Optional[str]]]],
        embed_dims: int,
        n_pos_groups: int,
        positional_embed_map: Dict[str, int],
        projection_multiplier: float = 4.0,
    ):
        super().__init__()
        assert len(embedders) == len(kwargs_info), "embedders and kwargs_info must align"
        self.module_list = embedders
        self.kwargs_info = kwargs_info
        self.embed_dims = embed_dims
        self.positional_embed_map = positional_embed_map
        self.use_pos_embedding = "learned"

        # learned per-modality bias (n_pos_groups, embed_dims)
        self.idx_emb = mx.zeros((n_pos_groups, embed_dims))

        self.projection_nets = []
        for emb in embedders:
            pre_norm, ffn = _make_projection_net(
                input_dim=emb.embed_dim,
                output_dim=embed_dims,
                multiplier=projection_multiplier,
            )
            # Wrap in a tiny container so PT key indices `.0` / `.1` map cleanly.
            container = _ProjectionPair(pre_norm=pre_norm, ffn=ffn)
            self.projection_nets.append(container)

    def __call__(self, **inputs: mx.array) -> mx.array:
        tokens: List[mx.array] = []
        for i, (emb, kwargs_for_emb) in enumerate(zip(self.module_list, self.kwargs_info)):
            for kwarg_name, pos_group in kwargs_for_emb:
                if kwarg_name not in inputs:
                    raise KeyError(f"missing kwarg {kwarg_name!r} for embedder {i}")
                cond_token = emb(inputs[kwarg_name])
                # projection: pre_norm -> FFN
                proj = self.projection_nets[i]
                cond_token = proj(cond_token)
                if pos_group is not None:
                    pos_idx = self.positional_embed_map[pos_group]
                    cond_token = cond_token + self.idx_emb[pos_idx : pos_idx + 1, None]
                tokens.append(cond_token)
        return mx.concatenate(tokens, axis=1)


class _ProjectionPair(nn.Module):
    """Container so PT key `projection_nets.{i}.0.*` and `.{i}.1.*` map onto
    `pre_norm` and `ffn` respectively.

    We override __call__ to apply pre_norm then ffn.
    """

    def __init__(self, pre_norm: nn.LayerNorm, ffn: _LlamaFFN):
        super().__init__()
        # Use list-style children so MLX serialises as projection_nets.{i}.0 / .1.
        self.layers = [pre_norm, ffn]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[1](self.layers[0](x))


# ---------------------------------------------------------------------------
# Top-level ConditionEmbedder (handles npz loading; used by inference pipeline).
# ---------------------------------------------------------------------------


# Configuration of which kwargs go through which embedder. This mirrors the
# PT ``embedder_list`` from ``checkpoints/{ss,slat}_generator.yaml`` (gated HF
# repo, snapshot fetched on the inference Pod). Each embedder is invoked
# **once per kwarg entry**, so the SS fuser produces 6 token blocks
# (1370 + 1370 + 1370 + 1370 + 1024 + 1024 = 7528 at 1024 dim) and the SLAT
# fuser produces 4 (4 * 1370 = 5480).
#
# Pos groups are ``cropped`` (idx 0) and ``full`` (idx 1). Note that
# ``idx_emb`` is allocated with ``max_pos_idx + 1`` rows, so the npz stores
# 3 rows for the SS / SLAT fusers even though only 2 are read at runtime.
#
# ``ss_generator.yaml`` (3 modules):
#   module_list.0 : Dino  -> [(image, cropped), (rgb_image, full)]
#   module_list.1 : Dino  -> [(mask, cropped), (rgb_image_mask, full)]
#   module_list.2 : PointPatchEmbed (remap_output=linear)
#                            -> [(pointmap, cropped), (rgb_pointmap, full)]
#
# ``slat_generator.yaml`` (2 modules):
#   module_list.0 : Dino  -> [(image, cropped), (rgb_image, full)]
#   module_list.1 : Dino  -> [(mask, cropped), (rgb_image_mask, full)]
#
# kwarg names match those produced by
# ``inference_pipeline_pointmap.preprocess_image`` (see ``item`` dict).

_DEFAULT_KWARG_PLAN_3 = [
    [("image", "cropped"), ("rgb_image", "full")],
    [("mask", "cropped"), ("rgb_image_mask", "full")],
    [("pointmap", "cropped"), ("rgb_pointmap", "full")],
]
_DEFAULT_KWARG_PLAN_2 = [
    [("image", "cropped"), ("rgb_image", "full")],
    [("mask", "cropped"), ("rgb_image_mask", "full")],
]


class ConditionEmbedder(nn.Module):
    """Top-level container exposed by SPEC_EMBED.md.

    Loads either ss_embedder.npz (3 modalities incl. pointmap) or
    slat_embedder.npz (2 modalities, image-only).
    """

    def __init__(self, fuser: EmbedderFuser):
        super().__init__()
        self.fuser = fuser

    def __call__(self, **kwargs: mx.array) -> mx.array:
        return self.fuser(**kwargs)

    @property
    def embed_dim(self) -> int:
        return self.fuser.embed_dims

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def from_npz(cls, npz_path: str) -> "ConditionEmbedder":
        weights = mx.load(npz_path)
        # Detect modality count from module_list indices.
        module_indices = sorted({int(k.split(".")[1]) for k in weights.keys()
                                 if k.startswith("module_list.")})
        n_modules = len(module_indices)

        # PT slat_embedder uses `prenorm_features=True` for DINO so register tokens
        # are kept in the output (chunk size = 1+4+1369 = 1374). PT ss_embedder uses
        # the default (`prenorm_features=False`) so registers are dropped (chunk size
        # = 1+1369 = 1370). We detect via filename — slat_embedder.npz keeps registers.
        keep_reg = "slat_embedder" in os.path.basename(npz_path).lower()

        if n_modules == 3:
            kwargs_plan = _DEFAULT_KWARG_PLAN_3
            embedders = [
                ImageEmbedder(keep_register_tokens=keep_reg),  # module 0
                ImageEmbedder(keep_register_tokens=keep_reg),  # module 1
                PointPatchEmbedder(remap_output="linear"),  # module 2
            ]
        elif n_modules == 2:
            kwargs_plan = _DEFAULT_KWARG_PLAN_2
            embedders = [
                ImageEmbedder(keep_register_tokens=keep_reg),
                ImageEmbedder(keep_register_tokens=keep_reg),
            ]
        else:
            raise RuntimeError(
                f"Unexpected n_modules={n_modules} in {npz_path}; expected 2 or 3."
            )

        # Build positional_embed_map from kwargs_plan.
        positional_embed_map: Dict[str, int] = {}
        next_idx = 0
        for entries in kwargs_plan:
            for _, pos_group in entries:
                if pos_group is not None and pos_group not in positional_embed_map:
                    positional_embed_map[pos_group] = next_idx
                    next_idx += 1

        # idx_emb shape determines target embed_dims.
        idx_emb = weights["idx_emb"]
        embed_dims = int(idx_emb.shape[-1])

        fuser = EmbedderFuser(
            embedders=embedders,
            kwargs_info=kwargs_plan,
            embed_dims=embed_dims,
            n_pos_groups=int(idx_emb.shape[0]),
            positional_embed_map=positional_embed_map,
            projection_multiplier=4.0,
        )

        # Inject weights tensor-by-tensor.
        _load_into_fuser(fuser, weights)

        return cls(fuser)


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _set(model_attr_owner, attr_name: str, value: mx.array):
    """Set an attribute (parameter / buffer) on a module."""
    setattr(model_attr_owner, attr_name, value)


def _load_dino_block(blk: _DinoBlock, w: dict, prefix: str):
    blk.norm1.weight = w[f"{prefix}.norm1.weight"]
    blk.norm1.bias = w[f"{prefix}.norm1.bias"]
    blk.norm2.weight = w[f"{prefix}.norm2.weight"]
    blk.norm2.bias = w[f"{prefix}.norm2.bias"]
    blk.attn.qkv.weight = w[f"{prefix}.attn.qkv.weight"]
    blk.attn.qkv.bias = w[f"{prefix}.attn.qkv.bias"]
    blk.attn.proj.weight = w[f"{prefix}.attn.proj.weight"]
    blk.attn.proj.bias = w[f"{prefix}.attn.proj.bias"]
    blk.mlp.fc1.weight = w[f"{prefix}.mlp.fc1.weight"]
    blk.mlp.fc1.bias = w[f"{prefix}.mlp.fc1.bias"]
    blk.mlp.fc2.weight = w[f"{prefix}.mlp.fc2.weight"]
    blk.mlp.fc2.bias = w[f"{prefix}.mlp.fc2.bias"]
    blk.ls1.gamma = w[f"{prefix}.ls1.gamma"]
    blk.ls2.gamma = w[f"{prefix}.ls2.gamma"]


def _load_dino_vit(vit: DinoViT, w: dict, prefix: str):
    # patch_embed
    vit.patch_embed.proj.weight = w[f"{prefix}.patch_embed.proj.weight"]
    vit.patch_embed.proj.bias = w[f"{prefix}.patch_embed.proj.bias"]
    # tokens
    vit.cls_token = w[f"{prefix}.cls_token"]
    vit.mask_token = w[f"{prefix}.mask_token"]
    vit.register_tokens = w[f"{prefix}.register_tokens"]
    vit.pos_embed = w[f"{prefix}.pos_embed"]
    # blocks
    for i, blk in enumerate(vit.blocks):
        _load_dino_block(blk, w, f"{prefix}.blocks.{i}")
    # final norm
    vit.norm.weight = w[f"{prefix}.norm.weight"]
    vit.norm.bias = w[f"{prefix}.norm.bias"]


def _load_point_embed(pe: PointPatchEmbedder, w: dict, prefix: str):
    pe.point_proj.weight = w[f"{prefix}.point_proj.weight"]
    pe.point_proj.bias = w[f"{prefix}.point_proj.bias"]
    pe.invalid_xyz_token = w[f"{prefix}.invalid_xyz_token"]
    pe.cls_token = w[f"{prefix}.cls_token"]
    pe.pos_embed = w[f"{prefix}.pos_embed"]
    pe.pos_embed_window = w[f"{prefix}.pos_embed_window"]
    for i, blk in enumerate(pe.blocks):
        # PointPatchEmbed blocks are timm-style without LayerScale.
        bp = f"{prefix}.blocks.{i}"
        blk.norm1.weight = w[f"{bp}.norm1.weight"]
        blk.norm1.bias = w[f"{bp}.norm1.bias"]
        blk.norm2.weight = w[f"{bp}.norm2.weight"]
        blk.norm2.bias = w[f"{bp}.norm2.bias"]
        blk.attn.qkv.weight = w[f"{bp}.attn.qkv.weight"]
        blk.attn.qkv.bias = w[f"{bp}.attn.qkv.bias"]
        blk.attn.proj.weight = w[f"{bp}.attn.proj.weight"]
        blk.attn.proj.bias = w[f"{bp}.attn.proj.bias"]
        blk.mlp.fc1.weight = w[f"{bp}.mlp.fc1.weight"]
        blk.mlp.fc1.bias = w[f"{bp}.mlp.fc1.bias"]
        blk.mlp.fc2.weight = w[f"{bp}.mlp.fc2.weight"]
        blk.mlp.fc2.bias = w[f"{bp}.mlp.fc2.bias"]


def _load_into_fuser(fuser: EmbedderFuser, w: dict):
    # idx_emb
    fuser.idx_emb = w["idx_emb"]
    # per-module
    for i, embedder in enumerate(fuser.module_list):
        if isinstance(embedder, ImageEmbedder):
            _load_dino_vit(embedder.backbone, w, f"module_list.{i}.backbone")
        elif isinstance(embedder, PointPatchEmbedder):
            _load_point_embed(embedder, w, f"module_list.{i}")
        else:
            raise TypeError(f"Unknown embedder type {type(embedder).__name__}")
    # projection_nets
    for i, proj in enumerate(fuser.projection_nets):
        proj.layers[0].weight = w[f"projection_nets.{i}.0.weight"]
        proj.layers[0].bias = w[f"projection_nets.{i}.0.bias"]
        proj.layers[1].w1.weight = w[f"projection_nets.{i}.1.w1.weight"]
        proj.layers[1].w2.weight = w[f"projection_nets.{i}.1.w2.weight"]
        proj.layers[1].w3.weight = w[f"projection_nets.{i}.1.w3.weight"]
