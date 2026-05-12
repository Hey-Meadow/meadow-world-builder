"""RAFT optical flow — MLX port (inference only, BasicRAFT only).

Weights are expected to be the converted `raft.npz` produced by
``scripts/convert_weights.py``. Conv2d weights are stored in MLX (OHWI) layout.

Public API:
    RAFT() -> module
    RAFT.load_weights(npz_path)
    RAFT(image1, image2, iters=12) -> (flow_low, flow_up)

Tensor conventions
------------------
All inputs/outputs follow MLX NHWC ordering. Images are float32 in [-1, 1]
(call sites should `2*img/255 - 1` before passing in).
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def coords_grid(batch: int, ht: int, wd: int) -> mx.array:
    """(B, H, W, 2) grid of pixel coordinates -> (x, y) on last axis."""
    ys, xs = mx.meshgrid(mx.arange(ht), mx.arange(wd), indexing="ij")
    g = mx.stack([xs, ys], axis=-1).astype(mx.float32)  # (H, W, 2)
    g = mx.broadcast_to(g[None], (batch, ht, wd, 2))
    return g


def bilinear_sample_nhwc(img: mx.array, coords: mx.array) -> mx.array:
    """Bilinear sample img at pixel coordinates.

    img:    (B, H, W, C)
    coords: (B, Hq, Wq, 2)  last dim = (x, y) in pixel units (align_corners=True)
    out:    (B, Hq, Wq, C)

    Semantics match PyTorch ``F.grid_sample(align_corners=True,
    padding_mode='zeros')``: each of the 4 bilinear neighbours contributes
    0 when its integer index is outside ``[0, H-1]`` × ``[0, W-1]``.
    """
    B, H, W, C = img.shape
    _, Hq, Wq, _ = coords.shape

    x = coords[..., 0]
    y = coords[..., 1]

    x0 = mx.floor(x)
    y0 = mx.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # per-corner validity (each integer neighbour must be inside the image)
    x0_v = (x0 >= 0) & (x0 <= W - 1)
    x1_v = (x1 >= 0) & (x1 <= W - 1)
    y0_v = (y0 >= 0) & (y0 <= H - 1)
    y1_v = (y1 >= 0) & (y1 <= H - 1)
    va = (x0_v & y0_v).astype(mx.float32)
    vb = (x0_v & y1_v).astype(mx.float32)
    vc = (x1_v & y0_v).astype(mx.float32)
    vd = (x1_v & y1_v).astype(mx.float32)

    x0c = mx.clip(x0, 0, W - 1).astype(mx.int32)
    x1c = mx.clip(x1, 0, W - 1).astype(mx.int32)
    y0c = mx.clip(y0, 0, H - 1).astype(mx.int32)
    y1c = mx.clip(y1, 0, H - 1).astype(mx.int32)

    b_idx = mx.arange(B).reshape(B, 1, 1)
    b_idx = mx.broadcast_to(b_idx, (B, Hq, Wq))

    Ia = img[b_idx, y0c, x0c, :]
    Ib = img[b_idx, y1c, x0c, :]
    Ic = img[b_idx, y0c, x1c, :]
    Id = img[b_idx, y1c, x1c, :]

    out = (Ia * (wa * va)[..., None]
           + Ib * (wb * vb)[..., None]
           + Ic * (wc * vc)[..., None]
           + Id * (wd * vd)[..., None])
    return out


def _avg_pool2d(x: mx.array, k: int = 2, stride: int = 2) -> mx.array:
    """Channels-last average pooling; treat each channel independently.

    Matches PyTorch ``F.avg_pool2d(x, k, stride=stride)`` with default
    ``ceil_mode=False`` and no padding — i.e. drops the trailing rows/cols
    that don't fit.
    """
    B, H, W, C = x.shape
    oh = (H - k) // stride + 1
    ow = (W - k) // stride + 1
    Hc = oh * stride  # truncate to multiple of stride
    Wc = ow * stride
    x = x[:, :Hc, :Wc, :]
    x = x.reshape(B, oh, stride, ow, stride, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    x = x.reshape(B, oh, ow, k * k, C).mean(axis=3)
    return x


def _interp_nhwc(x: mx.array, scale_factor: int) -> mx.array:
    """align_corners=True bilinear upsample by integer scale_factor (NHWC)."""
    B, H, W, C = x.shape
    if H == 1 or W == 1:  # fallback
        return mx.broadcast_to(x, (B, H * scale_factor, W * scale_factor, C))
    new_h = H * scale_factor
    new_w = W * scale_factor
    # generate output-pixel coords mapped back to input-pixel coords
    ys = mx.linspace(0, H - 1, new_h)
    xs = mx.linspace(0, W - 1, new_w)
    grid_y, grid_x = mx.meshgrid(ys, xs, indexing="ij")
    coords = mx.stack([grid_x, grid_y], axis=-1)  # (new_h, new_w, 2)
    coords = mx.broadcast_to(coords[None], (B, new_h, new_w, 2))
    return bilinear_sample_nhwc(x, coords)


# ---------------------------------------------------------------------------
# Norm helpers
# ---------------------------------------------------------------------------

class _InstanceNorm2dNHWC(nn.Module):
    """InstanceNorm over (H, W) for an NHWC tensor."""
    def __init__(self, num_features: int, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C). per-(B, C) statistics over H, W
        mean = x.mean(axis=(1, 2), keepdims=True)
        var = x.var(axis=(1, 2), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class _BatchNorm2dEval(nn.Module):
    """BatchNorm2d in eval mode (uses running stats)."""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        inv = self.weight / mx.sqrt(self.running_var + self.eps)
        return (x - self.running_mean) * inv + self.bias


def _make_norm(norm_fn: str, num_features: int):
    if norm_fn == "instance":
        return _InstanceNorm2dNHWC(num_features, affine=False)
    if norm_fn == "batch":
        return _BatchNorm2dEval(num_features)
    if norm_fn == "group":
        return nn.GroupNorm(num_groups=num_features // 8, dims=num_features, pytorch_compatible=True)
    if norm_fn == "none":
        return _Identity()
    raise ValueError(norm_fn)


class _Identity(nn.Module):
    def __call__(self, x):
        return x


# ---------------------------------------------------------------------------
# RAFT pieces
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.norm1 = _make_norm(norm_fn, planes)
        self.norm2 = _make_norm(norm_fn, planes)
        self.stride = stride
        if stride != 1:
            self.norm3 = _make_norm(norm_fn, planes)
            self.downsample_conv = nn.Conv2d(in_planes, planes, 1, stride=stride)
        else:
            self.norm3 = None
            self.downsample_conv = None

    def __call__(self, x):
        y = nn.relu(self.norm1(self.conv1(x)))
        y = nn.relu(self.norm2(self.conv2(y)))
        if self.downsample_conv is not None:
            x = self.norm3(self.downsample_conv(x))
        return nn.relu(x + y)


class BasicEncoder(nn.Module):
    """RAFT feature/context encoder. Returns features at 1/8 resolution."""
    def __init__(self, output_dim: int = 128, norm_fn: str = "batch"):
        super().__init__()
        self.norm_fn = norm_fn
        self.norm1 = _make_norm(norm_fn, 64)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)

        # Build layers; layers are nn.Sequential of 2 ResidualBlocks
        l1a = ResidualBlock(64, 64, norm_fn, stride=1)
        l1b = ResidualBlock(64, 64, norm_fn, stride=1)
        self.layer1 = nn.Sequential(l1a, l1b)

        l2a = ResidualBlock(64, 96, norm_fn, stride=2)
        l2b = ResidualBlock(96, 96, norm_fn, stride=1)
        self.layer2 = nn.Sequential(l2a, l2b)

        l3a = ResidualBlock(96, 128, norm_fn, stride=2)
        l3b = ResidualBlock(128, 128, norm_fn, stride=1)
        self.layer3 = nn.Sequential(l3a, l3b)

        self.conv2 = nn.Conv2d(128, output_dim, 1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
# Correlation pyramid
# ---------------------------------------------------------------------------

class CorrBlock:
    def __init__(self, fmap1: mx.array, fmap2: mx.array,
                 num_levels: int = 4, radius: int = 4):
        # fmap1/fmap2: (B, H, W, C)
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid: list[mx.array] = []

        B, H, W, C = fmap1.shape
        f1 = fmap1.reshape(B, H * W, C)
        f2 = fmap2.reshape(B, H * W, C)
        corr = (f1 @ f2.transpose(0, 2, 1)) / math.sqrt(C)  # (B, H*W, H*W)
        # store as (B*H*W, H, W, 1) for pooling/sampling
        corr = corr.reshape(B * H * W, H, W, 1)
        self.corr_pyramid.append(corr)
        for _ in range(num_levels - 1):
            corr = _avg_pool2d(corr, 2, 2)
            self.corr_pyramid.append(corr)
        self.B = B
        self.H = H
        self.W = W

    def __call__(self, coords: mx.array) -> mx.array:
        """coords: (B, H, W, 2) — pixel coords at fmap resolution.
        Returns (B, H, W, num_levels*(2r+1)^2)."""
        r = self.radius
        B, H, W, _ = coords.shape
        out_pyramid = []
        # Build delta grid (2r+1, 2r+1, 2).
        # PT mirrors a quirk: `torch.meshgrid(dy, dx)` with default 'ij'
        # indexing returns (grid_dy_first, grid_dx_second) and stacks them
        # with last-dim order (dy, dx). The bilinear sampler then treats
        # position 0 as "xgrid" -> W axis. We replicate that ordering
        # bit-for-bit so weights load identically.
        rs = mx.arange(-r, r + 1).astype(mx.float32)
        gA, gB = mx.meshgrid(rs, rs, indexing="ij")  # gA varies along axis 0
        delta = mx.stack([gA, gB], axis=-1)  # last dim: (dy, dx) — PT-compat
        K = (2 * r + 1)

        for i, corr in enumerate(self.corr_pyramid):
            # centroid: scale coords by 1/2**i, expand to delta grid
            centroid = (coords / (2 ** i)).reshape(B * H * W, 1, 1, 2)
            centroid = mx.broadcast_to(centroid, (B * H * W, K, K, 2))
            sample_coords = centroid + delta[None]  # (B*H*W, K, K, 2)
            sampled = bilinear_sample_nhwc(corr, sample_coords)  # (B*H*W, K, K, 1)
            sampled = sampled.reshape(B, H, W, K * K)
            out_pyramid.append(sampled)
        return mx.concatenate(out_pyramid, axis=-1)


# ---------------------------------------------------------------------------
# Update block
# ---------------------------------------------------------------------------

class FlowHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def __call__(self, x):
        return self.conv2(nn.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 320):
        super().__init__()
        d = hidden_dim + input_dim
        h = hidden_dim
        self.convz1 = nn.Conv2d(d, h, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(d, h, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(d, h, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(d, h, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(d, h, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(d, h, (5, 1), padding=(2, 0))

    def __call__(self, h, x):
        hx = mx.concatenate([h, x], axis=-1)
        z = mx.sigmoid(self.convz1(hx))
        r = mx.sigmoid(self.convr1(hx))
        q = mx.tanh(self.convq1(mx.concatenate([r * h, x], axis=-1)))
        h = (1 - z) * h + z * q

        hx = mx.concatenate([h, x], axis=-1)
        z = mx.sigmoid(self.convz2(hx))
        r = mx.sigmoid(self.convr2(hx))
        q = mx.tanh(self.convq2(mx.concatenate([r * h, x], axis=-1)))
        h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels: int = 4, corr_radius: int = 4):
        super().__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def __call__(self, flow, corr):
        cor = nn.relu(self.convc1(corr))
        cor = nn.relu(self.convc2(cor))
        flo = nn.relu(self.convf1(flow))
        flo = nn.relu(self.convf2(flo))
        cor_flo = mx.concatenate([cor, flo], axis=-1)
        out = nn.relu(self.conv(cor_flo))
        return mx.concatenate([out, flow], axis=-1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 128,
                 corr_levels: int = 4, corr_radius: int = 4):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        # Convex upsampling mask
        self.mask_conv0 = nn.Conv2d(128, 256, 3, padding=1)
        self.mask_conv2 = nn.Conv2d(256, 64 * 9, 1)

    def __call__(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = mx.concatenate([inp, motion_features], axis=-1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask_conv2(nn.relu(self.mask_conv0(net)))
        return net, mask, delta_flow


# ---------------------------------------------------------------------------
# RAFT
# ---------------------------------------------------------------------------

class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance")
        self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim,
                                 norm_fn="batch")
        self.update_block = BasicUpdateBlock(self.hidden_dim,
                                             corr_levels=self.corr_levels,
                                             corr_radius=self.corr_radius)

    # ---- weight loading ------------------------------------------------
    def load_npz(self, npz_path: str | Path):
        """Load weights from converted raft.npz."""
        data = np.load(str(npz_path))
        flat = {k: mx.array(data[k]) for k in data.files}
        mapping = self._pt_to_module_map()
        # cross-check
        missing = [k for k in mapping if mapping[k] not in flat]
        if missing:
            raise RuntimeError(f"Missing keys in npz: {missing[:6]}")

        loaded = 0
        for pt_key, npz_key in mapping.items():
            self._set_param(pt_key, flat[npz_key])
            loaded += 1
        # propagate to running stats for BatchNorm in cnet
        for pt_key in self._pt_to_module_map(include_bn_stats=True):
            mapped = self._pt_to_module_map(include_bn_stats=True)[pt_key]
            if mapped in flat and pt_key not in mapping:
                self._set_param(pt_key, flat[mapped])
        return loaded

    def _set_param(self, dotted: str, value: mx.array):
        parts = dotted.split(".")
        obj = self
        for p in parts[:-1]:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    @staticmethod
    def _pt_to_module_map(include_bn_stats: bool = True) -> dict[str, str]:
        """Maps internal module attribute paths -> npz keys."""
        # All keys in raft.npz follow PT naming (after `module.` strip).
        # We rename only the special cases:
        #   downsample.0 -> downsample_conv
        #   downsample.1 -> norm3
        #   mask.0       -> mask_conv0
        #   mask.2       -> mask_conv2
        m: dict[str, str] = {}

        def add(internal: str, npz: str):
            m[internal] = npz

        # ---- fnet / cnet encoders ----
        for enc in ("fnet", "cnet"):
            add(f"{enc}.conv1.weight", f"{enc}.conv1.weight")
            add(f"{enc}.conv1.bias",   f"{enc}.conv1.bias")
            add(f"{enc}.conv2.weight", f"{enc}.conv2.weight")
            add(f"{enc}.conv2.bias",   f"{enc}.conv2.bias")
            # norm1 — instance/batch
            if enc == "cnet":  # batch
                add(f"{enc}.norm1.weight",       f"{enc}.norm1.weight")
                add(f"{enc}.norm1.bias",         f"{enc}.norm1.bias")
                if include_bn_stats:
                    add(f"{enc}.norm1.running_mean", f"{enc}.norm1.running_mean")
                    add(f"{enc}.norm1.running_var",  f"{enc}.norm1.running_var")
            for li, layer_name in enumerate(["layer1", "layer2", "layer3"], start=1):
                for bi in (0, 1):
                    pre_int = f"{enc}.{layer_name}.layers.{bi}"
                    pre_pt = f"{enc}.{layer_name}.{bi}"
                    add(f"{pre_int}.conv1.weight", f"{pre_pt}.conv1.weight")
                    add(f"{pre_int}.conv1.bias",   f"{pre_pt}.conv1.bias")
                    add(f"{pre_int}.conv2.weight", f"{pre_pt}.conv2.weight")
                    add(f"{pre_int}.conv2.bias",   f"{pre_pt}.conv2.bias")
                    if enc == "cnet":
                        add(f"{pre_int}.norm1.weight", f"{pre_pt}.norm1.weight")
                        add(f"{pre_int}.norm1.bias",   f"{pre_pt}.norm1.bias")
                        add(f"{pre_int}.norm2.weight", f"{pre_pt}.norm2.weight")
                        add(f"{pre_int}.norm2.bias",   f"{pre_pt}.norm2.bias")
                        if include_bn_stats:
                            add(f"{pre_int}.norm1.running_mean", f"{pre_pt}.norm1.running_mean")
                            add(f"{pre_int}.norm1.running_var",  f"{pre_pt}.norm1.running_var")
                            add(f"{pre_int}.norm2.running_mean", f"{pre_pt}.norm2.running_mean")
                            add(f"{pre_int}.norm2.running_var",  f"{pre_pt}.norm2.running_var")
                    # downsample only on block 0 of layer2/layer3
                    if bi == 0 and layer_name in ("layer2", "layer3"):
                        add(f"{pre_int}.downsample_conv.weight", f"{pre_pt}.downsample.0.weight")
                        add(f"{pre_int}.downsample_conv.bias",   f"{pre_pt}.downsample.0.bias")
                        if enc == "cnet":
                            add(f"{pre_int}.norm3.weight", f"{pre_pt}.downsample.1.weight")
                            add(f"{pre_int}.norm3.bias",   f"{pre_pt}.downsample.1.bias")
                            if include_bn_stats:
                                add(f"{pre_int}.norm3.running_mean", f"{pre_pt}.downsample.1.running_mean")
                                add(f"{pre_int}.norm3.running_var",  f"{pre_pt}.downsample.1.running_var")

        # ---- update_block ----
        # encoder convs
        ub = "update_block"
        add(f"{ub}.encoder.convc1.weight", f"{ub}.encoder.convc1.weight")
        add(f"{ub}.encoder.convc1.bias",   f"{ub}.encoder.convc1.bias")
        add(f"{ub}.encoder.convc2.weight", f"{ub}.encoder.convc2.weight")
        add(f"{ub}.encoder.convc2.bias",   f"{ub}.encoder.convc2.bias")
        add(f"{ub}.encoder.convf1.weight", f"{ub}.encoder.convf1.weight")
        add(f"{ub}.encoder.convf1.bias",   f"{ub}.encoder.convf1.bias")
        add(f"{ub}.encoder.convf2.weight", f"{ub}.encoder.convf2.weight")
        add(f"{ub}.encoder.convf2.bias",   f"{ub}.encoder.convf2.bias")
        add(f"{ub}.encoder.conv.weight",   f"{ub}.encoder.conv.weight")
        add(f"{ub}.encoder.conv.bias",     f"{ub}.encoder.conv.bias")
        # gru
        for cn in ("convz1", "convr1", "convq1", "convz2", "convr2", "convq2"):
            add(f"{ub}.gru.{cn}.weight", f"{ub}.gru.{cn}.weight")
            add(f"{ub}.gru.{cn}.bias",   f"{ub}.gru.{cn}.bias")
        # flow_head
        add(f"{ub}.flow_head.conv1.weight", f"{ub}.flow_head.conv1.weight")
        add(f"{ub}.flow_head.conv1.bias",   f"{ub}.flow_head.conv1.bias")
        add(f"{ub}.flow_head.conv2.weight", f"{ub}.flow_head.conv2.weight")
        add(f"{ub}.flow_head.conv2.bias",   f"{ub}.flow_head.conv2.bias")
        # mask
        add(f"{ub}.mask_conv0.weight", f"{ub}.mask.0.weight")
        add(f"{ub}.mask_conv0.bias",   f"{ub}.mask.0.bias")
        add(f"{ub}.mask_conv2.weight", f"{ub}.mask.2.weight")
        add(f"{ub}.mask_conv2.bias",   f"{ub}.mask.2.bias")

        return m

    # ---- forward -------------------------------------------------------
    def initialize_flow(self, img: mx.array):
        B, H, W, _ = img.shape
        coords0 = coords_grid(B, H // 8, W // 8)
        coords1 = coords_grid(B, H // 8, W // 8)
        return coords0, coords1

    def upsample_flow(self, flow: mx.array, mask: mx.array) -> mx.array:
        """flow: (B, H, W, 2), mask: (B, H, W, 64*9). Upsample by 8."""
        B, H, W, _ = flow.shape
        # mask reshape to (B, H, W, 9, 8, 8)
        mask = mask.reshape(B, H, W, 9, 8, 8)
        mask = mx.softmax(mask, axis=3)

        # F.unfold(8*flow, k=3, pad=1) -> NHWC: pad and gather 3x3
        f = 8 * flow  # (B, H, W, 2)
        f_pad = mx.pad(f, [(0, 0), (1, 1), (1, 1), (0, 0)])
        # gather 9 shifted versions
        patches = []
        for dy in range(3):
            for dx in range(3):
                patches.append(f_pad[:, dy:dy + H, dx:dx + W, :])
        # (B, H, W, 9, 2)
        up_flow = mx.stack(patches, axis=3)
        # (B, H, W, 9, 8, 8, 2) by broadcasting; mask (B,H,W,9,8,8)
        up_flow = up_flow[:, :, :, :, None, None, :]
        out = (mask[..., None] * up_flow).sum(axis=3)  # (B, H, W, 8, 8, 2)
        # rearrange to (B, H*8, W*8, 2)
        out = out.transpose(0, 1, 3, 2, 4, 5)  # (B, H, 8, W, 8, 2)
        out = out.reshape(B, H * 8, W * 8, 2)
        return out

    def __call__(self, image1: mx.array, image2: mx.array,
                 iters: int = 20, flow_init: mx.array | None = None,
                 test_mode: bool = True):
        # Concatenate so InstanceNorm uses both frames (matches PT impl that
        # stacks then splits — InstanceNorm is per-(B, C), so equivalent).
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        corr_fn = CorrBlock(fmap1, fmap2,
                            num_levels=self.corr_levels, radius=self.corr_radius)

        cnet = self.cnet(image1)
        # split last axis into [hdim | cdim]
        net, inp = cnet[..., :self.hidden_dim], cnet[..., self.hidden_dim:]
        net = mx.tanh(net)
        inp = nn.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_up = None
        for _ in range(iters):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_up


# ---------------------------------------------------------------------------
# Input padding (replicate edges so H/W divisible by 8)
# ---------------------------------------------------------------------------

class InputPadder:
    """Replicate-pad so H, W are multiples of 8. Works on NHWC."""
    def __init__(self, dims: Sequence[int], mode: str = "sintel"):
        self.ht, self.wd = dims[1], dims[2]  # NHWC
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = (pad_ht // 2, pad_ht - pad_ht // 2,
                         pad_wd // 2, pad_wd - pad_wd // 2)
        else:
            self._pad = (0, pad_ht, pad_wd // 2, pad_wd - pad_wd // 2)

    def pad(self, *xs):
        out = []
        pt, pb, pl, pr = self._pad
        for x in xs:
            # replicate by edge-repeat: slice borders and concat
            if pt:
                top = mx.broadcast_to(x[:, :1, :, :], (x.shape[0], pt, x.shape[2], x.shape[3]))
                x = mx.concatenate([top, x], axis=1)
            if pb:
                bot = mx.broadcast_to(x[:, -1:, :, :], (x.shape[0], pb, x.shape[2], x.shape[3]))
                x = mx.concatenate([x, bot], axis=1)
            if pl:
                lft = mx.broadcast_to(x[:, :, :1, :], (x.shape[0], x.shape[1], pl, x.shape[3]))
                x = mx.concatenate([lft, x], axis=2)
            if pr:
                rgt = mx.broadcast_to(x[:, :, -1:, :], (x.shape[0], x.shape[1], pr, x.shape[3]))
                x = mx.concatenate([x, rgt], axis=2)
            out.append(x)
        return out

    def unpad(self, x):
        pt, pb, pl, pr = self._pad
        H, W = x.shape[1], x.shape[2]
        return x[:, pt:H - pb if pb else H, pl:W - pr if pr else W, :]
