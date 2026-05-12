"""Modulated deformable convolution v2 — MLX reference impl.

Matches the semantics of ``torchvision.ops.deform_conv2d(x, offset, weight,
bias, stride, padding, dilation, mask)`` for ``mask != None`` (i.e. DCNv2).

NHWC convention:
    x:      (B, H, W, Cin)
    offset: (B, OH, OW, 2 * deform_groups * kH * kW)
            -- last-dim layout matches PT: (g, kh, kw, [y,x]) flattened with
            innermost = (y, x). Group/(kh,kw) order follows PT.
    mask:   (B, OH, OW, deform_groups * kH * kW)
            -- already sigmoid'd by caller
    weight: (Cout, kH, kW, Cin)   (MLX layout)
    bias:   (Cout,) or None

This is a *reference* implementation; it materialises the sampling grid in
memory. For inference at video resolution (≤480p), peak memory is fine.
"""
from __future__ import annotations
from typing import Tuple
import mlx.core as mx

from .raft import bilinear_sample_nhwc


def _ensure_pair(v) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    return (int(v[0]), int(v[1]))


def modulated_deform_conv2d(
    x: mx.array,            # (B, H, W, Cin)
    offset: mx.array,       # (B, OH, OW, 2*dg*kH*kW)
    mask: mx.array,         # (B, OH, OW, dg*kH*kW)
    weight: mx.array,       # (Cout, kH, kW, Cin)
    bias: mx.array | None,  # (Cout,)
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
    dilation: int | Tuple[int, int] = 1,
    deform_groups: int = 1,
) -> mx.array:
    """Returns (B, OH, OW, Cout)."""
    B, H, W, Cin = x.shape
    Cout, kH, kW, _ = weight.shape
    sH, sW = _ensure_pair(stride)
    pH, pW = _ensure_pair(padding)
    dH, dW = _ensure_pair(dilation)
    OH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    OW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # 1) build base sampling grid in (B, OH, OW, kH*kW, 2)
    #    base_y = oh*sH - pH + kh*dH, base_x = ow*sW - pW + kw*dW
    oh = mx.arange(OH).astype(mx.float32) * sH - pH
    ow = mx.arange(OW).astype(mx.float32) * sW - pW
    kh = mx.arange(kH).astype(mx.float32) * dH
    kw = mx.arange(kW).astype(mx.float32) * dW
    # (OH, OW, kH, kW)
    base_y = oh[:, None, None, None] + kh[None, None, :, None]
    base_x = ow[None, :, None, None] + kw[None, None, None, :]
    base_y = mx.broadcast_to(base_y, (OH, OW, kH, kW))
    base_x = mx.broadcast_to(base_x, (OH, OW, kH, kW))

    # offset layout (PT torchvision deform_conv2d): along the channel dim
    # the order is (group, kh, kw, [y, x]) with [y, x] innermost. So a
    # reshape into (dg, kH, kW, 2) lays it out correctly.
    offset = offset.reshape(B, OH, OW, deform_groups, kH, kW, 2)
    off_y = offset[..., 0]   # (B, OH, OW, dg, kH, kW)
    off_x = offset[..., 1]
    mask = mask.reshape(B, OH, OW, deform_groups, kH, kW)

    # final sample coords per (B, OH, OW, dg, kH, kW)
    sy = off_y + base_y[None, :, :, None, :, :]
    sx = off_x + base_x[None, :, :, None, :, :]
    # flatten (dg, kH, kW) -> grid points: (B, OH, OW, dg*kH*kW)
    sy = sy.reshape(B, OH, OW, deform_groups * kH * kW)
    sx = sx.reshape(B, OH, OW, deform_groups * kH * kW)

    # 2) For each deform_group, we need to sample only its slice of channels.
    #    Cin must be divisible by deform_groups; channels per group = Cin // dg.
    cpg = Cin // deform_groups

    # Sample once per (group, kh, kw). We'll iterate over deform_groups to
    # keep memory bounded. For each group g:
    #   coords_g: (B, OH*OW*kH*kW, 2)
    #   sampled_g: (B, OH*OW*kH*kW, cpg)
    sampled_per_g = []
    for g in range(deform_groups):
        sy_g = sy.reshape(B, OH, OW, deform_groups, kH * kW)[:, :, :, g, :]   # (B, OH, OW, kH*kW)
        sx_g = sx.reshape(B, OH, OW, deform_groups, kH * kW)[:, :, :, g, :]
        coords = mx.stack([sx_g, sy_g], axis=-1)  # (B, OH, OW, kH*kW, 2)
        # flatten kernel dim into spatial query
        coords_flat = coords.reshape(B, OH, OW * (kH * kW), 2)
        # x slice: (B, H, W, cpg) — pick this group's channels
        x_g = x[..., g * cpg:(g + 1) * cpg]
        s = bilinear_sample_nhwc(x_g, coords_flat)  # (B, OH, OW*kH*kW, cpg)
        s = s.reshape(B, OH, OW, kH * kW, cpg)
        sampled_per_g.append(s)
    # (B, OH, OW, kH*kW, dg, cpg) -> (B, OH, OW, kH*kW, Cin)
    sampled = mx.stack(sampled_per_g, axis=-2)  # (B, OH, OW, kH*kW, dg, cpg)
    sampled = sampled.reshape(B, OH, OW, kH * kW, Cin)

    # 3) modulate by mask: each (kh, kw, dg) tile scales its channels.
    #    mask is (B, OH, OW, dg, kH*kW). We want (B, OH, OW, kH*kW, Cin)
    mask_kk = mask.transpose(0, 1, 2, 4, 5, 3)  # (B, OH, OW, kH, kW, dg)
    mask_kk = mask_kk.reshape(B, OH, OW, kH * kW, deform_groups)
    # expand to channels: repeat each dg entry cpg times along last axis
    mask_full = mx.repeat(mask_kk, cpg, axis=-1)  # (B, OH, OW, kH*kW, Cin)
    sampled = sampled * mask_full

    # 4) apply conv weights. weight: (Cout, kH, kW, Cin) -> (Cout, kH*kW*Cin)
    w_flat = weight.reshape(Cout, kH * kW * Cin)
    s_flat = sampled.reshape(B * OH * OW, kH * kW * Cin)
    out = s_flat @ w_flat.transpose(1, 0)   # (B*OH*OW, Cout)
    out = out.reshape(B, OH, OW, Cout)
    if bias is not None:
        out = out + bias
    return out
