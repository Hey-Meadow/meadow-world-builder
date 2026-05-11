"""Tier-1 quality gate for the CPU gsplat wrapper.

Three checks:
1. shape / finiteness: 1000 random Gaussians, 224x224, identity extrinsic.
2. speed: report sec/render (informational, no hard threshold).
3. parity: compare against a direct gsplat-style reference built from the
   same _torch_impl primitives. Max abs diff must be < 1e-3.

Note: upstream's full ``DecoderSplattingGSPlat.rendering_fn`` depends on the
real gsplat CUDA tile kernel and cannot run on this machine (no CUDA toolkit
on Apple Silicon). For the parity check we therefore build the reference
from the SAME pure-Python primitives gsplat ships (``_torch_impl``) — this is
mathematically the same projection upstream uses and gives a meaningful
self-consistency check of our rasterizer.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from meadow_sb.models.rasterizer import Gaussians, GsplatRasterizer


def _synth_gaussians(N: int, seed: int = 0) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    xyz = torch.randn(1, N, 3, generator=g) * 0.4
    xyz[..., 2] = xyz[..., 2].abs() + 2.0  # in front of camera
    scale = torch.ones(1, N, 3) * 0.03
    rotation = torch.zeros(1, N, 4)
    rotation[..., 0] = 1.0  # identity quat
    opacity = torch.full((1, N, 1), 0.5)
    # zero-degree SH: features (1, N, 3, 1). Set to constant RGB.
    features = torch.rand(1, N, 3, 1, generator=g) - 0.5
    return Gaussians(xyz=xyz, scale=scale, rotation=rotation,
                     opacity=opacity, features=features)


def _synth_cam(B: int = 1, V: int = 1, H: int = 224, W: int = 224):
    extr = torch.eye(4).reshape(1, 1, 4, 4).expand(B, V, 4, 4).contiguous()
    # Normalised intrinsics: fx/W = 0.7, fy/H = 0.7, cx/W = 0.5, cy/H = 0.5.
    Kn = torch.zeros(B, V, 3, 3)
    Kn[..., 0, 0] = 0.7
    Kn[..., 1, 1] = 0.7
    Kn[..., 0, 2] = 0.5
    Kn[..., 1, 2] = 0.5
    Kn[..., 2, 2] = 1.0
    return extr, Kn


# --------------------------------------------------------------------------- #
# Test 1: shape & finiteness
# --------------------------------------------------------------------------- #

def test_render_shape_and_finite():
    H, W = 224, 224
    g = _synth_gaussians(1000)
    extr, Kn = _synth_cam(1, 1, H, W)
    r = GsplatRasterizer(background=(0.5, 0.5, 0.5))
    img = r.render(g, extr, Kn, (H, W))
    assert img.shape == (1, 1, 3, H, W), img.shape
    assert torch.isfinite(img).all()
    # Should also be within valid colour range after upstream clamp.
    assert (img.min() >= 0.0) and (img.max() <= 1.0)


# --------------------------------------------------------------------------- #
# Test 2: speed (informational)
# --------------------------------------------------------------------------- #

def test_render_speed_report(capsys):
    H, W = 224, 224
    g = _synth_gaussians(1000)
    extr, Kn = _synth_cam(1, 1, H, W)
    r = GsplatRasterizer(background=(0.0, 0.0, 0.0))
    # warmup
    _ = r.render(g, extr, Kn, (H, W))
    t0 = time.perf_counter()
    n_iter = 3
    for _ in range(n_iter):
        _ = r.render(g, extr, Kn, (H, W))
    dt = (time.perf_counter() - t0) / n_iter
    print(f"[rasterizer] 1000 gaussians @ 224x224 CPU = {dt*1000:.1f} ms/render")
    # Soft sanity: under 60 s/render.
    assert dt < 60.0


# --------------------------------------------------------------------------- #
# Test 3: parity vs reference path through the SAME _torch_impl primitives
# --------------------------------------------------------------------------- #

def _reference_render(gauss: Gaussians, extr, Kn, H, W, background):
    """Reference path: directly call _torch_impl primitives the same way the
    rasterizer does, but with the simplest possible compositor inlined here
    (no chunking). Should agree with the production path to floating-point
    rounding error.

    This is a self-consistency check: it shares the projection + SH eval +
    compositing math with the production path. Its purpose is to catch
    regressions in how we wire those primitives together, not to certify
    against a black-box upstream binary.
    """
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
        _spherical_harmonics,
    )

    B, V = extr.shape[:2]
    w2c = torch.linalg.inv(extr.float())
    K_px = Kn.clone()
    K_px[..., 0, :] = K_px[..., 0, :] * W
    K_px[..., 1, :] = K_px[..., 1, :] * H

    covars, _ = _quat_scale_to_covar_preci(gauss.rotation, gauss.scale, True, False, triu=False)
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        gauss.xyz, covars, w2c, K_px, W, H,
        eps2d=0.3, near_plane=0.1, far_plane=1000.0,
    )

    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]
    cam_pos = -torch.einsum("bvji,bvj->bvi", R, t)
    dirs = gauss.xyz.unsqueeze(1) - cam_pos.unsqueeze(2)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    feat = gauss.features.permute(0, 1, 3, 2).contiguous()
    K_sh = feat.shape[-2]
    sh_degree = int(math.sqrt(K_sh)) - 1
    sh = feat.unsqueeze(1).expand(B, V, -1, K_sh, 3)
    colors = torch.clamp(_spherical_harmonics(sh_degree, dirs, sh) + 0.5, min=0.0)

    bg = torch.tensor(background, dtype=torch.float32)
    op = gauss.opacity[..., 0]

    out = torch.zeros(B, V, 3, H, W)
    for b in range(B):
        for v in range(V):
            m2 = means2d[b, v]; cn = conics[b, v]; cl = colors[b, v]
            opv = op[b]; rd = radii[b, v]; dp = depths[b, v]
            alive = (rd > 0).all(dim=-1) & (dp > 0.1) & (dp < 1000.0)
            m2 = m2[alive]; cn = cn[alive]; cl = cl[alive]
            opv = opv[alive]; rd = rd[alive]; dp = dp[alive]
            order = torch.argsort(dp)
            m2 = m2[order]; cn = cn[order]; cl = cl[order]
            opv = opv[order]; rd = rd[order]
            ys, xs = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing="ij",
            )
            px = xs + 0.5; py = ys + 0.5
            rgb = torch.zeros(H, W, 3)
            T = torch.ones(H, W)
            for i in range(m2.shape[0]):
                rx = float(rd[i, 0]); ry = float(rd[i, 1])
                if rx <= 0 or ry <= 0:
                    continue
                cx = float(m2[i, 0]); cy = float(m2[i, 1])
                dx = px - cx; dy = py - cy
                a, bb, c = cn[i, 0], cn[i, 1], cn[i, 2]
                power = -0.5 * (a * dx * dx + c * dy * dy) - bb * dx * dy
                power = torch.clamp(power, max=0.0)
                alpha = torch.clamp(opv[i] * torch.exp(power), max=0.999)
                mask = (alpha > (1.0 / 255.0)) & (T > 1e-4)
                w = (alpha * T) * mask
                rgb += cl[i].view(1, 1, 3) * w.unsqueeze(-1)
                T = T * (1.0 - alpha * mask)
            rgb = rgb + bg.view(1, 1, 3) * T.unsqueeze(-1)
            rgb = rgb.clamp(0.0, 1.0)
            out[b, v] = rgb.permute(2, 0, 1)
    return out


def test_render_matches_reference():
    H, W = 64, 64  # small for the slow reference loop
    g = _synth_gaussians(200, seed=42)
    extr, Kn = _synth_cam(1, 1, H, W)
    r = GsplatRasterizer(background=(0.2, 0.3, 0.4),
                         opacity_threshold=0.0,   # disable prune for exact compare
                         radius_clip=0.0)
    out = r.render(g, extr, Kn, (H, W))
    ref = _reference_render(g, extr, Kn, H, W, (0.2, 0.3, 0.4))
    diff = (out - ref).abs().max().item()
    print(f"[rasterizer] max abs diff vs reference = {diff:.3e}")
    assert diff < 1e-3, f"max abs diff {diff} exceeds 1e-3"
