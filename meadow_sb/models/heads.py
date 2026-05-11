"""YoNoSplat output heads, MLX port.

Five head modules sitting on top of the three sub-decoders + the backbone:

    GaussianHead    Linear(1024, 539)        per-token Gaussian param vector
    PointHead       Linear(1024, 147)        per-token raw point projection
                                             (147 = 3 * points_per_token, where
                                              points_per_token = 7^2 for
                                              upscale_token_ratio=2 and
                                              gaussians_per_axis=14)
    CameraHead      ResConv x2 + AvgPool +   per-view 12-dim pose
                    MLP + fc_rot/fc_t        (9 rotation + 3 translation,
                                              dim=512 input)
    IntrinsicHead   MLP(1024, 1024, 2)       fx, fy
    RgbEmbed        Conv2d(3, 2048, 7, 7)    rgb pixel embedding fed back into
                    + LayerNorm              the gaussian-stream tokens.

State-dict prefixes (encoder.* in the upstream checkpoint):

    encoder.gaussian_head.proj.{weight,bias}        (539, 1024) / (539,)
    encoder.point_head.proj.{weight,bias}           (147, 1024) / (147,)
    encoder.camera_head.res_conv.{0,1}.res_conv{1,2,3}.{weight,bias}
                                                    (512, 512) / (512,)
    encoder.camera_head.more_mlps.{0,2}.{weight,bias}
                                                    (512, 512) / (512,)
    encoder.camera_head.fc_t.{weight,bias}          (3, 512)   / (3,)
    encoder.camera_head.fc_rot.{weight,bias}        (9, 512)   / (9,)
    encoder.backbone.intrinsic_head.fc{1,2}.{weight,bias}
                                                    (1024,1024)+(2,1024)
    encoder.rgb_embed.proj.{weight,bias}            (2048, 3, 7, 7) / (2048,)
    encoder.rgb_embed.norm.{weight,bias}            (2048,) / (2048,)

References:
    /tmp/yonosplat_inspect/src/model/encoder/encoder_yonosplat.py
    /tmp/yonosplat_inspect/src/model/encoder/layers/camera_head.py
    /tmp/yonosplat_inspect/src/model/encoder/backbone/dinov2/layers/patch_embed.py
"""
from __future__ import annotations

from typing import Mapping

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers


def _load_linear(layer: nn.Linear, weight: mx.array, bias: mx.array) -> None:
    """Copy PT-style (out, in) weight + (out,) bias into an mlx.nn.Linear."""
    layer.weight = weight
    layer.bias = bias


# ---------------------------------------------------------------------------
# Tiny heads (single Linear)


class GaussianHead(nn.Module):
    """Per-token Gaussian parameter projection.

    The 539-dim output is a single Linear; the *upscale / reshape* to a full
    Gaussian field happens **outside** this head (pixel_shuffle in
    `LinearPts3d.forward`). The head itself is just `Linear(1024, 539)`.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 539) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class PointHead(nn.Module):
    """Per-token raw point projection.

    output_dim is 147 = 3 (xyz) * points_per_token (= 7^2 for the default
    upscale_token_ratio=2 / gaussians_per_axis=14 config). The contract sheet
    summarises this as "1024 -> 3" because the per-pixel xyz pops out after a
    pixel-shuffle later in the pipeline.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 147) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


# ---------------------------------------------------------------------------
# CameraHead — most internal structure


class _ResConvBlock(nn.Module):
    """1x1 residual block, but implemented as Linear (matches upstream)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # in == out → no skip projection; upstream uses nn.Identity().
        self.res_conv1 = nn.Linear(dim, dim, bias=True)
        self.res_conv2 = nn.Linear(dim, dim, bias=True)
        self.res_conv3 = nn.Linear(dim, dim, bias=True)

    def __call__(self, res: mx.array) -> mx.array:
        x = nn.relu(self.res_conv1(res))
        x = nn.relu(self.res_conv2(x))
        x = nn.relu(self.res_conv3(x))
        return res + x


class CameraHead(nn.Module):
    """Camera pose head: (B, hw, 512) feature -> (B, 12) = 9 rot + 3 t.

    Pipeline:
        feat                 (B, hw, dim)
        -> 2x _ResConvBlock  (B, hw, dim)         # token-wise Linears
        -> reshape           (B, dim, patch_h, patch_w)
        -> adaptive avg pool to (B, dim, 1, 1)
        -> flatten           (B, dim)
        -> more_mlps         (B, dim)
        -> fc_rot / fc_t     (B, 9) + (B, 3) concatenated to (B, 12)

    This head **does not** apply the SVD orthogonalisation / 4x4 assembly that
    upstream's `convert_pose_to_4x4` does; that's deferred to the encoder
    glue (Agent C / E territory) because it depends on the SO(3) projection
    library we settle on for MLX.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.res_conv = [_ResConvBlock(dim), _ResConvBlock(dim)]
        # `more_mlps` is `Sequential(Linear, ReLU, Linear, ReLU)` upstream;
        # we keep the same numbered indexing (`0`, `2`) so the weight loader
        # can be a near-1:1 key map.
        self.more_mlps_0 = nn.Linear(dim, dim, bias=True)
        self.more_mlps_2 = nn.Linear(dim, dim, bias=True)
        self.fc_t = nn.Linear(dim, 3, bias=True)
        self.fc_rot = nn.Linear(dim, 9, bias=True)

    def __call__(
        self,
        feat: mx.array,
        patch_h: int,
        patch_w: int,
    ) -> mx.array:
        # feat: (BN, hw, dim)
        for block in self.res_conv:
            feat = block(feat)

        bn, hw, c = feat.shape
        assert hw == patch_h * patch_w, (
            f"CameraHead: hw={hw} != patch_h*patch_w={patch_h*patch_w}"
        )
        # Reshape (BN, hw, c) -> (BN, patch_h, patch_w, c) then avg-pool spatially.
        # MLX is channels-last by default which matches token-major naturally.
        feat = feat.reshape(bn, patch_h, patch_w, c)
        # Adaptive avg pool to 1x1 == mean over the spatial dims.
        feat = feat.mean(axis=(1, 2))  # (BN, c)

        feat = nn.relu(self.more_mlps_0(feat))
        feat = nn.relu(self.more_mlps_2(feat))

        out_t = self.fc_t(feat)       # (BN, 3)
        out_r = self.fc_rot(feat)     # (BN, 9)
        return mx.concatenate([out_r, out_t], axis=-1)  # (BN, 12)


# ---------------------------------------------------------------------------
# IntrinsicHead — Mlp(1024, 1024, 2)


class IntrinsicHead(nn.Module):
    """Predict (fx, fy) from a 1024-d cls token.

    Architecture per upstream `backbone_local_global.py`:
        fc1: Linear(1024, 1024)
        ReLU
        fc2: Linear(1024, 2)
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 1024, out_dim: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# RgbEmbed — Conv2d patch embedding


class RgbEmbed(nn.Module):
    """Tiny PatchEmbed for the gaussian-decoder RGB injection path.

    Upstream uses dinov2.PatchEmbed with patch_size=7 (= 14 / upscale_token_ratio=2),
    embed_dim=2048, in_chans=3, and a final LayerNorm. Output is
    (B, num_patches, embed_dim) where num_patches = (224 / 7)^2 = 1024.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 2048,
        patch_size: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) -- mlx Conv2d is channels-last.

        Caller must transpose from PT-style (B, C, H, W) before calling, OR
        feed the data in channels-last layout to begin with. Helper
        `from_pt_image` does the transpose if you already have a PT-layout
        array.
        """
        x = self.proj(x)  # (B, H/p, W/p, embed_dim)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        x = self.norm(x)
        return x


def rgb_embed_from_pt(x_bchw: mx.array) -> mx.array:
    """Convenience: turn a PT-layout (B, C, H, W) array into MLX (B, H, W, C)."""
    return mx.transpose(x_bchw, (0, 2, 3, 1))


# ---------------------------------------------------------------------------
# Weight loading: PT state-dict -> MLX heads


def _pt_to_mx(pt_tensor) -> mx.array:
    """Detach a torch tensor and reify as mlx array via numpy (float32 only)."""
    import numpy as np

    return mx.array(np.ascontiguousarray(pt_tensor.detach().cpu().float().numpy()))


def load_gaussian_head(state_dict: Mapping[str, "object"], prefix: str = "encoder.gaussian_head") -> GaussianHead:
    head = GaussianHead()
    head.proj.weight = _pt_to_mx(state_dict[f"{prefix}.proj.weight"])
    head.proj.bias = _pt_to_mx(state_dict[f"{prefix}.proj.bias"])
    return head


def load_point_head(state_dict: Mapping[str, "object"], prefix: str = "encoder.point_head") -> PointHead:
    head = PointHead()
    head.proj.weight = _pt_to_mx(state_dict[f"{prefix}.proj.weight"])
    head.proj.bias = _pt_to_mx(state_dict[f"{prefix}.proj.bias"])
    return head


def load_camera_head(state_dict: Mapping[str, "object"], prefix: str = "encoder.camera_head") -> CameraHead:
    head = CameraHead(dim=512)
    for i in range(2):
        block = head.res_conv[i]
        for j in (1, 2, 3):
            lin: nn.Linear = getattr(block, f"res_conv{j}")
            lin.weight = _pt_to_mx(state_dict[f"{prefix}.res_conv.{i}.res_conv{j}.weight"])
            lin.bias = _pt_to_mx(state_dict[f"{prefix}.res_conv.{i}.res_conv{j}.bias"])
    head.more_mlps_0.weight = _pt_to_mx(state_dict[f"{prefix}.more_mlps.0.weight"])
    head.more_mlps_0.bias = _pt_to_mx(state_dict[f"{prefix}.more_mlps.0.bias"])
    head.more_mlps_2.weight = _pt_to_mx(state_dict[f"{prefix}.more_mlps.2.weight"])
    head.more_mlps_2.bias = _pt_to_mx(state_dict[f"{prefix}.more_mlps.2.bias"])
    head.fc_t.weight = _pt_to_mx(state_dict[f"{prefix}.fc_t.weight"])
    head.fc_t.bias = _pt_to_mx(state_dict[f"{prefix}.fc_t.bias"])
    head.fc_rot.weight = _pt_to_mx(state_dict[f"{prefix}.fc_rot.weight"])
    head.fc_rot.bias = _pt_to_mx(state_dict[f"{prefix}.fc_rot.bias"])
    return head


def load_intrinsic_head(
    state_dict: Mapping[str, "object"],
    prefix: str = "encoder.backbone.intrinsic_head",
) -> IntrinsicHead:
    head = IntrinsicHead()
    head.fc1.weight = _pt_to_mx(state_dict[f"{prefix}.fc1.weight"])
    head.fc1.bias = _pt_to_mx(state_dict[f"{prefix}.fc1.bias"])
    head.fc2.weight = _pt_to_mx(state_dict[f"{prefix}.fc2.weight"])
    head.fc2.bias = _pt_to_mx(state_dict[f"{prefix}.fc2.bias"])
    return head


def load_rgb_embed(state_dict: Mapping[str, "object"], prefix: str = "encoder.rgb_embed") -> RgbEmbed:
    head = RgbEmbed()
    # PT Conv2d weight: (out, in, kH, kW)
    # MLX Conv2d weight: (out, kH, kW, in)
    import numpy as np

    w_pt = state_dict[f"{prefix}.proj.weight"].detach().cpu().float().numpy()
    w_mx = np.transpose(w_pt, (0, 2, 3, 1))  # NCHW->NHWC for kernel
    head.proj.weight = mx.array(np.ascontiguousarray(w_mx))
    head.proj.bias = _pt_to_mx(state_dict[f"{prefix}.proj.bias"])
    head.norm.weight = _pt_to_mx(state_dict[f"{prefix}.norm.weight"])
    head.norm.bias = _pt_to_mx(state_dict[f"{prefix}.norm.bias"])
    return head


__all__ = [
    "GaussianHead",
    "PointHead",
    "CameraHead",
    "IntrinsicHead",
    "RgbEmbed",
    "rgb_embed_from_pt",
    "load_gaussian_head",
    "load_point_head",
    "load_camera_head",
    "load_intrinsic_head",
    "load_rgb_embed",
]
