"""MoGe (Monocular Geometry) MLX port — DINOv2 ViT-L/14 + Head.

Replaces the dummy synthetic pointmap in ``pipeline_mlx.py`` with real depth
predictions. Mirrors the PyTorch reference at
``moge.model.v1.MoGeModel`` with these adaptations:

- ``DINOv2VisionTransformerMoGe``: ``num_register_tokens=0`` variant of the
  existing lingbot DINOv2 port, with ``interpolate_offset=0.1`` (legacy
  kludge MoGe uses) and ``init_values=1.0`` LayerScale.
- ``MoGeHead``: 4 projection convs (1024 -> 512), 3 upsample blocks
  (ConvTranspose2d + Conv2d + N residual blocks), then 2 output blocks
  (3-channel xyz + 1-channel mask). UV channels are concatenated at every
  upsample stage and at the final output stage.
- ``recover_focal_shift``: numpy-only fallback uses the existing PT helper
  ``moge.utils.geometry_numpy.solve_optimal_focal_shift``. Cheap (downsample
  to 64x64), so we cross the MLX/numpy boundary once per inference.

Key design choice: we *do not* port `force_projection=True` (depth_to_points
re-projection). The downstream pipeline calls MoGe with
``force_projection=False`` per the PT wrapper at
``sam3d_objects/pipeline/depth_models/moge.py:25``. Output points are simply
shifted in z by the recovered shift.

Inputs: NHWC RGB image in [0,1] (float32). Outputs: pointmap (B, 3, H, W),
mask (B, 1, H, W), intrinsics (B, 3, 3) — all CPU-resident MLX arrays.

License: code structure mirrors Apache-2.0 DINOv2 + MoGe sources.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_MEAN = mx.array([0.485, 0.456, 0.406], dtype=mx.float32)
IMAGE_STD = mx.array([0.229, 0.224, 0.225], dtype=mx.float32)

# Token range for Ruicheng/moge-vitl: trained_area_range=[250000, 500000]
# divided by 14**2 = (1275, 2551).
DEFAULT_NUM_TOKENS_RANGE = (1275, 2551)


# ---------------------------------------------------------------------------
# DINOv2 ViT-L/14 (no register, MoGe variant)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Conv2d patch embedding with NHWC convention (MLX-native)."""

    def __init__(self, embed_dim: int = 1024, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, int, int]:
        # x: (B, H, W, 3)
        _, h, w, _ = x.shape
        x = self.proj(x)  # (B, ph, pw, C)
        _, ph, pw, c = x.shape
        return x.reshape(x.shape[0], ph * pw, c), ph, pw


class _Attention(nn.Module):
    """ViT attention with mx.fast.scaled_dot_product_attention."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float):
        super().__init__()
        self.gamma = mx.ones((dim,), dtype=mx.float32) * init_values

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 init_values: float, qkv_bias: bool = True,
                 layer_norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = _Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.ls1 = _LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim, init_values)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2MoGe(nn.Module):
    """DINOv2 ViT-L/14, MoGe variant (no register tokens, interpolate_offset=0.1)."""

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        patch_size: int = 14,
        img_size: int = 518,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        interpolate_offset: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.interpolate_offset = interpolate_offset

        base_grid = img_size // patch_size  # 37
        num_patches = base_grid * base_grid  # 1369
        self.patch_embed = _PatchEmbed(embed_dim, patch_size)
        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))

        self.blocks = [
            _Block(embed_dim, num_heads, mlp_ratio, init_values,
                   qkv_bias=True, layer_norm_eps=layer_norm_eps)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # Buffer-only (not used by MoGe, but present in state_dict)
        self.mask_token = mx.zeros((1, embed_dim))

    def _interpolate_pos_encoding(
        self, x: mx.array, ph: int, pw: int
    ) -> mx.array:
        npatch = x.shape[1] - 1
        n_base = self.pos_embed.shape[1] - 1
        base = int(math.sqrt(n_base))
        if npatch == n_base and ph == pw:
            return self.pos_embed.astype(x.dtype)
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:].reshape(1, base, base, self.embed_dim)
        # Use scale_factor with the +0.1 offset (matches MoGe DINOv2 path).
        sx = float(pw + self.interpolate_offset) / base
        sy = float(ph + self.interpolate_offset) / base
        patch_pos = nn.Upsample(
            scale_factor=(sy, sx), mode="cubic", align_corners=False
        )(patch_pos)
        # MLX Upsample with float scale_factor may produce ph or ph+1 due to round.
        # Trim to (ph, pw).
        patch_pos = patch_pos[:, :ph, :pw]
        patch_pos = patch_pos.reshape(1, ph * pw, self.embed_dim)
        return mx.concatenate([cls_pos, patch_pos], axis=1).astype(x.dtype)

    def _prepare_tokens(self, x: mx.array) -> mx.array:
        x, ph, pw = self.patch_embed(x)
        b = x.shape[0]
        cls = mx.broadcast_to(self.cls_token, (b, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)
        x = x + self._interpolate_pos_encoding(x, ph, pw)
        return x

    def get_intermediate_layers(
        self,
        x: mx.array,
        n: int = 4,
        return_class_token: bool = True,
        norm: bool = True,
    ) -> list:
        """Return last `n` block outputs.

        Each item is (patch_tokens, class_token), matching MoGe usage.
        """
        x = self._prepare_tokens(x)
        depth = len(self.blocks)
        wanted = list(range(depth - n, depth)) if isinstance(n, int) else list(n)
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in wanted:
                out = self.norm(x) if norm else x
                cls = out[:, 0]
                patches = out[:, 1:]
                outputs.append((patches, cls))
        return outputs


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def _replicate_pad2d(x: mx.array, pad: int) -> mx.array:
    """Replicate (edge) padding for NHWC arrays. Pad H and W by `pad` on each side."""
    if pad == 0:
        return x
    # Pad height: prepend `pad` copies of x[:, 0:1] and append `pad` copies of x[:, -1:].
    top = mx.broadcast_to(x[:, :1], (x.shape[0], pad, x.shape[2], x.shape[3]))
    bot = mx.broadcast_to(x[:, -1:], (x.shape[0], pad, x.shape[2], x.shape[3]))
    x = mx.concatenate([top, x, bot], axis=1)
    left = mx.broadcast_to(x[:, :, :1], (x.shape[0], x.shape[1], pad, x.shape[3]))
    right = mx.broadcast_to(x[:, :, -1:], (x.shape[0], x.shape[1], pad, x.shape[3]))
    return mx.concatenate([left, x, right], axis=2)


class _ReplicateConv2d(nn.Module):
    """Conv2d with replicate padding (NHWC). Padding mode='replicate'."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        x = _replicate_pad2d(x, self.padding)
        return self.conv(x)


def _normalized_view_plane_uv(width: int, height: int,
                              aspect_ratio: float = None) -> mx.array:
    """UV grid with diagonal normalization. Returns (H, W, 2)."""
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1.0 + aspect_ratio ** 2) ** 0.5
    span_y = 1.0 / (1.0 + aspect_ratio ** 2) ** 0.5
    u = mx.linspace(-span_x * (width - 1) / width,
                    span_x * (width - 1) / width, width)
    v = mx.linspace(-span_y * (height - 1) / height,
                    span_y * (height - 1) / height, height)
    # meshgrid 'xy': u broadcast over rows, v over cols.
    U, V = mx.meshgrid(u, v, indexing="xy")
    return mx.stack([U, V], axis=-1)  # (H, W, 2)


class _ResidualConvBlock(nn.Module):
    """ResidualConvBlock: GroupNorm -> ReLU -> Conv -> GN -> ReLU -> Conv. No bias-free."""

    def __init__(self, in_ch: int, out_ch: int, hidden_ch: int,
                 norm_groups_first: int = 1):
        super().__init__()
        # PT layout (mirrors moge.model.v1.ResidualConvBlock):
        #   layers[0] = GroupNorm(1, in_ch)
        #   layers[1] = ReLU (no params)
        #   layers[2] = Conv2d(in_ch, hidden_ch, 3, padding=1, replicate)
        #   layers[3] = GroupNorm(hidden_ch//32 (or 1), hidden_ch)
        #   layers[4] = ReLU
        #   layers[5] = Conv2d(hidden_ch, out_ch, 3, padding=1, replicate)
        self.gn1 = nn.GroupNorm(num_groups=1, dims=in_ch, pytorch_compatible=True)
        # Note: index 2 in PT Sequential
        self.conv1 = _ReplicateConv2d(in_ch, hidden_ch, 3, 1, 1)
        gn2_groups = hidden_ch // 32 if (hidden_ch // 32) > 0 else 1
        self.gn2 = nn.GroupNorm(num_groups=gn2_groups, dims=hidden_ch,
                                pytorch_compatible=True)
        self.conv2 = _ReplicateConv2d(hidden_ch, out_ch, 3, 1, 1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        else:
            self.skip = None

    def __call__(self, x: mx.array) -> mx.array:
        skip = x if self.skip is None else self.skip(x)
        h = self.gn1(x)
        h = nn.relu(h)
        h = self.conv1(h)
        h = self.gn2(h)
        h = nn.relu(h)
        h = self.conv2(h)
        return h + skip


class _UpsampleBlock(nn.Module):
    """ConvTranspose2d (k=2,s=2) -> Conv2d(replicate) -> N residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int = 1,
                 hidden_mult: int = 1):
        super().__init__()
        # in_ch already includes the +2 UV concat.
        self.conv_t = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ReplicateConv2d(out_ch, out_ch, 3, 1, 1)
        # ResidualConvBlock(in_ch=out_ch, out_ch=out_ch,
        #                   hidden=hidden_mult*out_ch). Note: PT default for
        # hidden_channels is in_channels, but MoGe explicitly passes
        # `dim_times_res_block_hidden * out_ch`, so hidden = mult * out_ch.
        self.res_blocks = [
            _ResidualConvBlock(out_ch, out_ch, hidden_mult * out_ch)
            for _ in range(num_res_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_t(x)
        x = self.conv(x)
        for blk in self.res_blocks:
            x = blk(x)
        return x


class _OutputBlock(nn.Module):
    """Conv (3x3 replicate) -> 0..N residual blocks -> ReLU -> Conv (1x1) projecting to dim_out."""

    def __init__(self, in_ch: int, out_ch: int, last_conv_channels: int = 32,
                 last_res_blocks: int = 0, last_conv_size: int = 1,
                 hidden_mult: int = 1):
        super().__init__()
        self.conv1 = _ReplicateConv2d(in_ch, last_conv_channels, 3, 1, 1)
        self.res_blocks = [
            _ResidualConvBlock(last_conv_channels, last_conv_channels,
                               hidden_mult * last_conv_channels)
            for _ in range(last_res_blocks)
        ]
        # final conv (kernel_size=last_conv_size, padding=last_conv_size//2 replicate).
        pad = last_conv_size // 2
        if pad == 0:
            self.conv_out = nn.Conv2d(last_conv_channels, out_ch,
                                      kernel_size=last_conv_size,
                                      padding=0)
            self._needs_pad = False
        else:
            self.conv_out = _ReplicateConv2d(last_conv_channels, out_ch,
                                             last_conv_size, 1, pad)
            self._needs_pad = True

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        for blk in self.res_blocks:
            x = blk(x)
        x = nn.relu(x)
        x = self.conv_out(x)
        return x


class MoGeHead(nn.Module):
    """MoGe Head: project intermediate ViT features -> upsample 8x -> 2 outputs."""

    def __init__(
        self,
        num_features: int = 4,
        dim_in: int = 1024,
        dim_proj: int = 512,
        dim_upsample: Iterable[int] = (256, 128, 64),
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        last_res_blocks: int = 0,
        num_res_blocks: int = 2,
        hidden_mult: int = 2,
    ):
        super().__init__()
        dim_upsample = list(dim_upsample)
        self.projects = [
            nn.Conv2d(dim_in, dim_proj, kernel_size=1, padding=0)
            for _ in range(num_features)
        ]
        in_chs = [dim_proj] + dim_upsample[:-1]
        self.upsample_blocks = [
            _UpsampleBlock(in_ch + 2, out_ch, num_res_blocks, hidden_mult)
            for in_ch, out_ch in zip(in_chs, dim_upsample)
        ]
        # Two output blocks: 3-channel xyz + 1-channel mask.
        last_in = dim_upsample[-1] + 2
        self.output_block = [
            _OutputBlock(last_in, dim_out, last_conv_channels,
                         last_res_blocks, last_conv_size, hidden_mult)
            for dim_out in (3, 1)
        ]

    def __call__(
        self, intermediate_features: list, image_h: int, image_w: int
    ) -> Tuple[mx.array, mx.array]:
        """intermediate_features: list of (patch_tokens, class_token).

        patch_tokens: (B, ph*pw, C). class_token: (B, C). class_token unused.

        Returns:
            points: (B, 3, image_h, image_w)
            mask:   (B, 1, image_h, image_w)
        """
        # Reshape patch tokens to (B, C, ph, pw) -> NHWC for MLX.
        patch_h = image_h // 14
        patch_w = image_w // 14
        # Project each layer and sum.
        x = None
        for proj, (patches, _cls) in zip(self.projects, intermediate_features):
            b, n, c = patches.shape
            # (B, ph, pw, C) NHWC layout.
            feat = patches.reshape(b, patch_h, patch_w, c)
            feat = proj(feat)
            x = feat if x is None else x + feat  # (B, ph, pw, dim_proj)

        # Upsample stages: (ph, pw) -> 2x -> 4x -> 8x.
        for blk in self.upsample_blocks:
            b, h, w, c = x.shape
            uv = _normalized_view_plane_uv(w, h, aspect_ratio=image_w / image_h)
            uv = mx.broadcast_to(uv[None], (b, h, w, 2))
            x = mx.concatenate([x, uv], axis=-1)
            x = blk(x)

        # Final upsample to image size (bilinear) + UV concat.
        # MLX nn.Upsample only supports scale_factor; use float-scale to map
        # (8 * patch_h, 8 * patch_w) -> (image_h, image_w).
        sy = image_h / x.shape[1]
        sx = image_w / x.shape[2]
        x = nn.Upsample(scale_factor=(sy, sx), mode="linear",
                        align_corners=False)(x)
        x = x[:, :image_h, :image_w]
        b, h, w, c = x.shape
        uv = _normalized_view_plane_uv(w, h, aspect_ratio=image_w / image_h)
        uv = mx.broadcast_to(uv[None], (b, h, w, 2))
        x = mx.concatenate([x, uv], axis=-1)

        # Two output heads.
        out_xyz_nhwc = self.output_block[0](x)  # (B, H, W, 3)
        out_mask_nhwc = self.output_block[1](x)  # (B, H, W, 1)

        # Convert to (B, C, H, W) channels-first to match PT `MoGeModel`.
        points = mx.transpose(out_xyz_nhwc, (0, 3, 1, 2))
        mask = mx.transpose(out_mask_nhwc, (0, 3, 1, 2))
        return points, mask


# ---------------------------------------------------------------------------
# MoGeModel
# ---------------------------------------------------------------------------


class MoGeModel(nn.Module):
    """DINOv2 ViT-L/14 + MoGeHead. Produces pointmap, mask, intrinsics."""

    def __init__(
        self,
        encoder_kwargs: dict = None,
        head_kwargs: dict = None,
        num_tokens_range: Tuple[int, int] = DEFAULT_NUM_TOKENS_RANGE,
        mask_threshold: float = 0.5,
        remap_output: str = "exp",
    ):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        head_kwargs = head_kwargs or {}
        self.backbone = DINOv2MoGe(**encoder_kwargs)
        self.head = MoGeHead(**head_kwargs)
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold
        self.remap_output = remap_output
        self.image_mean = IMAGE_MEAN
        self.image_std = IMAGE_STD
        self.intermediate_layers = head_kwargs.get("num_features", 4)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "Ruicheng/moge-vitl",
        cache_dir: Optional[str] = None,
        force_convert: bool = False,
    ) -> "MoGeModel":
        """Load MoGe from HF hub (cache to local npz on first call)."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "weights",
                "moge",
            )
        os.makedirs(cache_dir, exist_ok=True)
        npz_path = os.path.join(cache_dir, "moge_vitl.npz")

        # Single model config for both cached and fresh paths (matches
        # Ruicheng/moge-vitl checkpoint).
        model_kwargs = dict(
            encoder_kwargs=dict(
                embed_dim=1024, depth=24, num_heads=16,
                patch_size=14, img_size=518, mlp_ratio=4.0,
                init_values=1.0, interpolate_offset=0.1,
            ),
            head_kwargs=dict(
                num_features=4, dim_in=1024, dim_proj=512,
                dim_upsample=(256, 128, 64),
                last_conv_channels=32, last_conv_size=1,
                last_res_blocks=0, num_res_blocks=2,
                hidden_mult=2,
            ),
            remap_output="exp",
            num_tokens_range=DEFAULT_NUM_TOKENS_RANGE,
        )
        if os.path.exists(npz_path) and not force_convert:
            print(f"[moge_mlx] loading cached npz: {npz_path}")
            weights = mx.load(npz_path)
        else:
            print(f"[moge_mlx] no cached npz at {npz_path}, converting from PT...")
            weights = _convert_pt_checkpoint(pretrained_model_name_or_path,
                                             npz_path)
        model = cls(**model_kwargs)
        _load_weights_into(model, weights)
        return model

    def _normalize(self, image: mx.array) -> mx.array:
        # image: (B, H, W, 3) float in [0, 1].
        return (image - self.image_mean) / self.image_std

    def _resize_bilinear_nhwc(self, x: mx.array, h: int, w: int) -> mx.array:
        """Bilinear resize NHWC array to (h, w)."""
        cur_h, cur_w = x.shape[1], x.shape[2]
        if cur_h == h and cur_w == w:
            return x
        sy = h / cur_h
        sx = w / cur_w
        x = nn.Upsample(scale_factor=(sy, sx), mode="linear",
                        align_corners=False)(x)
        return x[:, :h, :w]

    def _resize_bicubic_nhwc(self, x: mx.array, h: int, w: int) -> mx.array:
        cur_h, cur_w = x.shape[1], x.shape[2]
        if cur_h == h and cur_w == w:
            return x
        sy = h / cur_h
        sx = w / cur_w
        x = nn.Upsample(scale_factor=(sy, sx), mode="cubic",
                        align_corners=False)(x)
        return x[:, :h, :w]

    def __call__(
        self,
        image: mx.array,
        num_tokens: Optional[int] = None,
        resolution_level: int = 9,
        apply_mask: bool = True,
    ) -> dict:
        """Run MoGe inference on a single (or batched) image.

        Args:
            image: (B, H, W, 3) float NHWC RGB in [0, 1].
            num_tokens: override token budget. Default: derived from
                ``resolution_level`` and ``num_tokens_range``.
            resolution_level: integer 0-9 controlling output detail.
            apply_mask: if True, set pointmap pixels with mask<thresh to inf.

        Returns:
            dict with keys ``points`` (B, 3, H, W), ``mask`` (B, 1, H, W),
            ``mask_prob`` (B, 1, H, W), ``depth`` (B, H, W), ``intrinsics``
            (B, 3, 3).
        """
        if image.ndim == 3:
            image = image[None]
        B, original_h, original_w, _ = image.shape

        if num_tokens is None:
            min_t, max_t = self.num_tokens_range
            num_tokens = int(min_t + (resolution_level / 9.0) * (max_t - min_t))

        # 1) Resize to expected resolution.
        resize_factor = ((num_tokens * 14 ** 2) / (original_h * original_w)) ** 0.5
        resized_w = int(original_w * resize_factor)
        resized_h = int(original_h * resize_factor)
        # Match PT: resize bicubic + antialias. We use bicubic (no antialias in MLX).
        image_r = self._resize_bicubic_nhwc(image, resized_h, resized_w)

        # 2) Normalize ImageNet mean/std.
        image_r = self._normalize(image_r)

        # 3) Resize to multiples of 14 for ViT.
        image_14_h = (resized_h // 14) * 14
        image_14_w = (resized_w // 14) * 14
        image_14 = self._resize_bilinear_nhwc(image_r, image_14_h, image_14_w)

        # 4) Run ViT, get last 4 intermediate layers.
        n_layers = self.intermediate_layers
        if not isinstance(n_layers, int):
            n_layers = len(n_layers)
        feats = self.backbone.get_intermediate_layers(
            image_14, n=n_layers, return_class_token=True, norm=True
        )

        # 5) Run head -> points (B, 3, H', W'), mask (B, 1, H', W').
        points, mask = self.head(feats, image_14_h, image_14_w)

        # 6) Resize back to original.
        # points/mask are (B, C, H, W). Convert to NHWC for resize.
        def _resize_chw(x_chw, h, w, mode="linear"):
            x_hwc = mx.transpose(x_chw, (0, 2, 3, 1))
            if mode == "linear":
                x_hwc = self._resize_bilinear_nhwc(x_hwc, h, w)
            else:
                x_hwc = self._resize_bicubic_nhwc(x_hwc, h, w)
            return mx.transpose(x_hwc, (0, 3, 1, 2))

        points = _resize_chw(points, original_h, original_w, mode="linear")
        mask = _resize_chw(mask, original_h, original_w, mode="linear")

        # 6.5) Apply remap_output (matches PT MoGeModel._remap_points). For
        # the released checkpoint this is 'exp': z = exp(z), xy = xy * z.
        if self.remap_output == "exp":
            xy = points[:, :2]
            z = mx.exp(points[:, 2:3])
            xy_remap = xy * z
            points = mx.concatenate([xy_remap, z], axis=1)
        elif self.remap_output == "sinh":
            points = mx.sinh(points)
        elif self.remap_output == "sinh_exp":
            xy = mx.sinh(points[:, :2])
            z = mx.exp(points[:, 2:3])
            points = mx.concatenate([xy, z], axis=1)
        elif self.remap_output in ("linear", True, False):
            pass
        else:
            raise ValueError(f"Invalid remap_output: {self.remap_output}")

        # 7) Solve focal+shift via numpy (downsampled to 64x64).
        mask_prob = mx.sigmoid(mask)
        # PT applies threshold on raw logits with mask_threshold=0.5.
        mask_binary = mask > self.mask_threshold

        focal, shift = _recover_focal_shift_numpy(
            points, mask_binary, downsample_size=(64, 64)
        )
        # focal, shift: shape (B,) MLX arrays.

        # 8) Apply shift to z channel only (force_projection=False path).
        # points: (B, 3, H, W). Add shift to z.
        z = points[:, 2:3] + shift.reshape(B, 1, 1, 1)
        points = mx.concatenate([points[:, :2], z], axis=1)
        depth = points[:, 2]

        # 9) Compute intrinsics.
        aspect_ratio = original_w / original_h
        # fx = focal/2 * sqrt(1+a^2)/a, fy = focal/2 * sqrt(1+a^2)
        sqrt_term = (1.0 + aspect_ratio ** 2) ** 0.5
        fx = focal / 2.0 * sqrt_term / aspect_ratio
        fy = focal / 2.0 * sqrt_term
        # Build (B, 3, 3): cx = cy = 0.5
        intr = mx.zeros((B, 3, 3), dtype=mx.float32)
        intr_np = np.zeros((B, 3, 3), dtype=np.float32)
        fx_np = np.asarray(fx)
        fy_np = np.asarray(fy)
        for i in range(B):
            intr_np[i, 0, 0] = float(fx_np[i])
            intr_np[i, 1, 1] = float(fy_np[i])
            intr_np[i, 0, 2] = 0.5
            intr_np[i, 1, 2] = 0.5
            intr_np[i, 2, 2] = 1.0
        intr = mx.array(intr_np)

        # 10) Apply mask: outside -> inf.
        if apply_mask:
            mask_b = mask_binary.astype(mx.float32)  # (B, 1, H, W)
            inf = mx.array(float("inf"), dtype=mx.float32)
            points = mx.where(
                mask_b > 0.5, points, mx.broadcast_to(inf, points.shape)
            )
            depth = mx.where(
                mask_b[:, 0] > 0.5, depth, mx.broadcast_to(inf, depth.shape)
            )

        return {
            "points": points,        # (B, 3, H, W)
            "mask": mask_binary,     # (B, 1, H, W) bool
            "mask_prob": mask_prob,  # (B, 1, H, W)
            "depth": depth,          # (B, H, W)
            "intrinsics": intr,      # (B, 3, 3)
            "shift": shift,          # (B,)
            "focal": focal,          # (B,)
        }


# ---------------------------------------------------------------------------
# Weight conversion from PT
# ---------------------------------------------------------------------------


def _convert_pt_checkpoint(repo_id_or_path: str, npz_out: str) -> dict:
    """Download MoGe model.pt, convert state_dict to MLX npz, return weights dict."""
    import torch
    from huggingface_hub import hf_hub_download

    p = Path(repo_id_or_path)
    if p.exists() and p.is_file():
        ckpt_path = str(p)
    else:
        print(f"[moge_mlx] downloading {repo_id_or_path}/model.pt ...")
        ckpt_path = hf_hub_download(
            repo_id=repo_id_or_path,
            repo_type="model",
            filename="model.pt",
        )
    print(f"[moge_mlx] loading PT checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"]
    out: dict[str, mx.array] = {}

    for k, v in state.items():
        # Skip image_mean / image_std — we hardcode them.
        if k in ("image_mean", "image_std"):
            continue
        # Drop register_tokens (MoGe variant has none, but safeguard).
        if k == "backbone.register_tokens":
            continue
        # Convert.
        new_key, new_val = _convert_tensor(k, v)
        if new_key is None:
            continue
        out[new_key] = new_val

    # Save once.
    print(f"[moge_mlx] saving npz: {npz_out} ({len(out)} arrays)")
    mx.save_safetensors(npz_out.replace(".npz", ".safetensors"), out) if False else mx.savez(npz_out, **out)
    return out


def _convert_tensor(key: str, t) -> Tuple[Optional[str], Optional[mx.array]]:
    """Map PT key -> MLX key, transpose tensor to MLX layout."""
    import torch

    arr = t.detach().cpu().to(torch.float32).numpy()
    # ConvTranspose2d weights: in upsample blocks, head.upsample_blocks.{i}.0.0.weight.
    # PT shape: (in, out, kH, kW) -> MLX: (out, kH, kW, in).
    if key.endswith(".weight") and arr.ndim == 4:
        # Determine if Conv2d or ConvTranspose2d.
        # Heuristic: head.upsample_blocks.{i}.0.0 = ConvTranspose2d (kernel_size=2, stride=2).
        is_conv_t = ".upsample_blocks." in key and key.endswith(".0.0.weight")
        if is_conv_t:
            arr = np.ascontiguousarray(arr.transpose(1, 2, 3, 0))
        else:
            arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))

    # Build new key.
    new_key = _remap_pt_key(key)
    if new_key is None:
        return None, None
    return new_key, mx.array(arr)


def _remap_pt_key(key: str) -> Optional[str]:
    """Map PT MoGeModel state_dict keys -> our MLX module keys.

    PT layout (sample):
        backbone.patch_embed.proj.weight     -> backbone.patch_embed.proj.weight
        backbone.cls_token                   -> backbone.cls_token
        backbone.pos_embed                   -> backbone.pos_embed
        backbone.mask_token                  -> backbone.mask_token
        backbone.norm.{weight,bias}          -> backbone.norm.{weight,bias}
        backbone.blocks.{i}.norm1.{w,b}      -> backbone.blocks.{i}.norm1.{w,b}
        backbone.blocks.{i}.attn.qkv.{w,b}   -> backbone.blocks.{i}.attn.qkv.{w,b}
        backbone.blocks.{i}.attn.proj.{w,b}  -> backbone.blocks.{i}.attn.proj.{w,b}
        backbone.blocks.{i}.ls1.gamma        -> backbone.blocks.{i}.ls1.gamma
        backbone.blocks.{i}.norm2.{w,b}      -> ...
        backbone.blocks.{i}.mlp.fc{1,2}.*    -> ...
        backbone.blocks.{i}.ls2.gamma        -> ...

        head.projects.{i}.{weight,bias}       -> head.projects.{i}.{weight,bias}

        head.upsample_blocks.{i}.0.0.weight   -> head.upsample_blocks.{i}.conv_t.weight
        head.upsample_blocks.{i}.0.1.weight   -> head.upsample_blocks.{i}.conv.conv.weight
        head.upsample_blocks.{i}.1.layers.0.weight -> head.upsample_blocks.{i}.res_blocks.0.gn1.weight
        head.upsample_blocks.{i}.1.layers.2.weight -> head.upsample_blocks.{i}.res_blocks.0.conv1.conv.weight
        head.upsample_blocks.{i}.1.layers.3.weight -> head.upsample_blocks.{i}.res_blocks.0.gn2.weight
        head.upsample_blocks.{i}.1.layers.5.weight -> head.upsample_blocks.{i}.res_blocks.0.conv2.conv.weight

        head.output_block.{0,1}.0.weight     -> head.output_block.{0,1}.conv1.conv.weight
        head.output_block.{0,1}.2.weight     -> head.output_block.{0,1}.conv_out.weight
            (conv_out has padding=0 since last_conv_size=1 -> no replicate wrap)
    """
    # Backbone passthrough.
    if key.startswith("backbone."):
        return key

    # Head projects.
    if key.startswith("head.projects."):
        return key

    # Head upsample blocks.
    if key.startswith("head.upsample_blocks."):
        # Parse: head.upsample_blocks.{i}.{branch}.{rest}
        parts = key.split(".")
        # parts[0]='head', [1]='upsample_blocks', [2]=i, [3]=branch, ...
        i = parts[2]
        branch = parts[3]
        if branch == "0":
            # 0.0 = conv_t (ConvTranspose2d), 0.1 = conv (Conv2d replicate)
            sub = parts[4]
            tail = ".".join(parts[5:])
            if sub == "0":
                return f"head.upsample_blocks.{i}.conv_t.{tail}"
            elif sub == "1":
                return f"head.upsample_blocks.{i}.conv.conv.{tail}"
        else:
            # Residual blocks live at indices >= 1.
            # PT wraps as nn.Sequential(_make_upsampler(...), *res_blocks).
            # head.upsample_blocks.{i}.{1+r}.layers.* is res_block[r].
            res_idx = int(branch) - 1
            sub = parts[4]
            if sub == "layers":
                layer_idx = int(parts[5])
                tail = ".".join(parts[6:])
                # PT layers index -> MLX submodule:
                #   0 -> gn1
                #   2 -> conv1.conv (the wrapped Conv2d inside _ReplicateConv2d)
                #   3 -> gn2
                #   5 -> conv2.conv
                mlx_sub = {0: "gn1", 2: "conv1.conv", 3: "gn2",
                           5: "conv2.conv"}.get(layer_idx)
                if mlx_sub is None:
                    return None
                return f"head.upsample_blocks.{i}.res_blocks.{res_idx}.{mlx_sub}.{tail}"
            elif sub == "skip_connection":
                tail = ".".join(parts[5:])
                # skip_connection only present when in_ch != out_ch.
                return f"head.upsample_blocks.{i}.res_blocks.{res_idx}.skip.{tail}"
        return None

    # Head output blocks.
    if key.startswith("head.output_block."):
        # Pattern: head.output_block.{i}.{seq_idx}.{rest}
        parts = key.split(".")
        # parts[0]='head', [1]='output_block', [2]=i, [3]=seq_idx
        i = parts[2]
        seq_idx = int(parts[3])
        tail = ".".join(parts[4:])
        # PT Sequential layout (num_res_blocks=0, last_res_blocks=0):
        #   0 -> Conv2d(dim_in, last_conv_channels=32, 3, replicate)
        #   1 -> ReLU (no params)
        #   2 -> Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size=1)
        if seq_idx == 0:
            return f"head.output_block.{i}.conv1.conv.{tail}"
        elif seq_idx == 2:
            # last_conv_size=1 -> no replicate wrap (pad=0).
            return f"head.output_block.{i}.conv_out.{tail}"
        return None

    return None


def _load_weights_into(model: nn.Module, weights: dict) -> None:
    """Load flat-key weights into model via mlx.nn.Module.load_weights."""
    items = []
    for k, v in weights.items():
        if not isinstance(v, mx.array):
            v = mx.array(np.asarray(v))
        items.append((k, v))
    # strict=False to skip image_mean/image_std (constants set in __init__).
    model.load_weights(items, strict=False)


# ---------------------------------------------------------------------------
# Numpy bridge for focal/shift recovery
# ---------------------------------------------------------------------------


def _recover_focal_shift_numpy(
    points: mx.array,
    mask: mx.array,
    downsample_size: Tuple[int, int] = (64, 64),
) -> Tuple[mx.array, mx.array]:
    """Closed-form solver for (focal, shift). Mirrors PT recover_focal_shift.

    points: (B, 3, H, W) MLX. mask: (B, 1, H, W) bool MLX.
    Returns (focal, shift) MLX arrays of shape (B,).
    """
    from moge.utils.geometry_numpy import solve_optimal_focal_shift  # type: ignore

    pts_np = np.asarray(points, dtype=np.float32)  # (B, 3, H, W)
    msk_np = np.asarray(mask, dtype=np.bool_)[:, 0]  # (B, H, W)
    B, _, H, W = pts_np.shape

    # Move channels last and downsample (nearest, single batch loop is fine).
    pts_hwc = np.transpose(pts_np, (0, 2, 3, 1))  # (B, H, W, 3)
    # Build UV for the FULL resolution (for downsampling).
    aspect = W / H
    span_x = aspect / (1.0 + aspect ** 2) ** 0.5
    span_y = 1.0 / (1.0 + aspect ** 2) ** 0.5
    u = np.linspace(-span_x * (W - 1) / W, span_x * (W - 1) / W, W,
                    dtype=np.float32)
    v = np.linspace(-span_y * (H - 1) / H, span_y * (H - 1) / H, H,
                    dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="xy")
    uv = np.stack([U, V], axis=-1)  # (H, W, 2)

    dh, dw = downsample_size
    # Nearest downsample using stride.
    sy = max(1, H // dh)
    sx = max(1, W // dw)
    pts_lr = pts_hwc[:, ::sy, ::sx][:, :dh, :dw]
    msk_lr = msk_np[:, ::sy, ::sx][:, :dh, :dw]
    uv_lr = uv[::sy, ::sx][:dh, :dw]

    focal = np.zeros((B,), dtype=np.float32)
    shift = np.zeros((B,), dtype=np.float32)
    for i in range(B):
        msk_i = msk_lr[i]
        if msk_i.sum() < 2:
            focal[i] = 1.0
            shift[i] = 0.0
            continue
        pts_i = pts_lr[i][msk_i]
        uv_i = uv_lr[msk_i]
        s_i, f_i = solve_optimal_focal_shift(uv_i, pts_i)
        focal[i] = float(f_i)
        shift[i] = float(s_i)

    return mx.array(focal), mx.array(shift)


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------


_MODEL_SINGLETON: Optional[MoGeModel] = None


def get_or_load_moge() -> MoGeModel:
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        t0 = time.time()
        _MODEL_SINGLETON = MoGeModel.from_pretrained()
        print(f"[moge_mlx] from_pretrained done in {time.time() - t0:.1f} s")
    return _MODEL_SINGLETON


def infer_pointmap(
    rgba_uint8: np.ndarray,
    pm_size: int = 256,
    model: Optional[MoGeModel] = None,
) -> Tuple[mx.array, mx.array]:
    """Convenience helper used by ``pipeline_mlx.py``.

    Args:
        rgba_uint8: (H, W, 4) uint8 RGBA. Alpha = object mask. RGB has been
            background-cleared (rembg already applied).
        pm_size: target pointmap H/W (PointPatchEmbed expects 256).
        model: optional pre-loaded MoGeModel (defaults to singleton).

    Returns:
        pointmap_b3hw: (1, 3, pm_size, pm_size) float MLX. NaN outside mask.
        rgb_pointmap_nhwc: same shape as the input pre-resize image.
    """
    if model is None:
        model = get_or_load_moge()

    rgb = rgba_uint8[:, :, :3].astype(np.float32) / 255.0
    image = mx.array(rgb[None])  # (1, H, W, 3)
    out = model(image, apply_mask=False)  # we'll mask manually below
    points = out["points"]  # (1, 3, H, W) — H,W matches input.
    mask_b = np.asarray(out["mask"])[0, 0]  # (H, W) bool

    # Convert to numpy for resize + masking.
    pts_np = np.asarray(points)[0]  # (3, H, W)
    H, W = pts_np.shape[1:]

    # Combine MoGe mask AND the input alpha mask (rembg from upstream).
    alpha = rgba_uint8[:, :, 3] > 0  # (H, W)
    final_mask = mask_b & alpha

    # Apply pytorch3d camera convention transform that PT pipeline does:
    # look_at_view_transform(eye=[0,0,-1], at=[0,0,0], up=[0,-1,0]) maps
    # (x, y, z) -> (-x, -y, z). Equivalently sign flip on x and y channels.
    pts_np[0] = -pts_np[0]
    pts_np[1] = -pts_np[1]

    # Apply mask: outside -> NaN.
    pts_np = pts_np.copy()
    pts_np[:, ~final_mask] = np.nan

    # Resize to (pm_size, pm_size) using nearest (matches PT
    # F.interpolate(mode="nearest") in compute_pointmap line 286).
    if H != pm_size or W != pm_size:
        from PIL import Image as _PILImage
        sy = pm_size / H
        sx = pm_size / W
        out_resized = np.empty((3, pm_size, pm_size), dtype=np.float32)
        for c in range(3):
            ch = pts_np[c]
            ch_pil = _PILImage.fromarray(ch)
            ch_pil = ch_pil.resize((pm_size, pm_size), _PILImage.NEAREST)
            out_resized[c] = np.asarray(ch_pil)
        pts_np = out_resized

    return mx.array(pts_np[None]), out


__all__ = [
    "MoGeModel",
    "DINOv2MoGe",
    "MoGeHead",
    "infer_pointmap",
    "get_or_load_moge",
]
