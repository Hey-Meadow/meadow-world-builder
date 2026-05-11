"""MLX port of YoNoSplat's GaussianAdapter (UnifiedGaussianAdapter variant).

Upstream: `src/model/encoder/common/gaussian_adapter.py::UnifiedGaussianAdapter`.

What this module does:
- Consumes the raw per-Gaussian tensor produced by the gaussian_head + the
  per-Gaussian opacity (split off from raw outside) + per-Gaussian means
  (from the point_head).
- Splits the raw vector into (scale, rotation, sh) channels, applies the
  proper activations (softplus*0.001 then clamp, quaternion normalize,
  SH mask), builds the world-space covariance and returns a `Gaussians`
  dataclass with the same field names as upstream.

Layout of the per-token 539-d vector (gaussian_head output) for the re10k
config (sh_degree=0, gaussians_per_axis=14, upscale_token_ratio=2,
num_surfaces=1):

    per_token_dim = (gaussians_per_axis / upscale_token_ratio)^2 * raw_gs_dim
                  = 7^2 * 11
                  = 49 * 11
                  = 539

    where raw_gs_dim = 1 + d_in = 1 + (7 + 3 * d_sh) = 1 + 10 = 11

    per-Gaussian 11-dim split:
        [0    : 1]   opacity_logit   (1)
        [1    : 4]   scale           (3)
        [4    : 8]   rotation        (4)  (i, j, k, r quaternion)
        [8    : 11]  sh              (3 * d_sh = 3)

The caller (encoder) splits `[opacity, raw10]` *before* invoking the adapter,
so this module sees:

    raw_gaussians: (..., 10)   # scale | rotation | sh
    opacities:     (..., 1)    # already sigmoided + pdf_to_opacity mapped
    means:         (..., 3)    # from point_head
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Gaussians:
    """MLX Gaussians dataclass — mirrors upstream
    `src/model/encoder/common/gaussian_adapter.py::Gaussians`.

    Field shapes (matching upstream after the broadcast in `.forward`):

        means        : (..., 3)
        covariances  : (..., 3, 3)
        scales       : (..., 3)
        rotations    : (..., 4)
        harmonics    : (..., 3, d_sh)
        opacities    : (...)        # last dim of `1` may be squeezed by caller
    """

    means: mx.array
    covariances: mx.array
    scales: mx.array
    rotations: mx.array
    harmonics: mx.array
    opacities: mx.array


@dataclass
class GaussianAdapterCfg:
    sh_degree: int                 # 0 for re10k
    gaussian_scale_min: float      # only used by the (non-unified) base path
    gaussian_scale_max: float      # only used by the (non-unified) base path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softplus(x: mx.array) -> mx.array:
    # Numerically stable softplus: log(1 + exp(x))
    # mlx has nn.softplus but operating directly keeps this module self-contained.
    return mx.logaddexp(x, mx.zeros_like(x))


def _quaternion_to_matrix(q: mx.array, eps: float = 1e-8) -> mx.array:
    """Replica of upstream `gaussians.quaternion_to_matrix`.

    Quaternion order is (i, j, k, r) -- scipy/xyzw convention.
    """
    # q shape: (..., 4)
    i = q[..., 0]
    j = q[..., 1]
    k = q[..., 2]
    r = q[..., 3]

    two_s = 2.0 / ((q * q).sum(axis=-1) + eps)  # (...)

    m00 = 1 - two_s * (j * j + k * k)
    m01 = two_s * (i * j - k * r)
    m02 = two_s * (i * k + j * r)
    m10 = two_s * (i * j + k * r)
    m11 = 1 - two_s * (i * i + k * k)
    m12 = two_s * (j * k - i * r)
    m20 = two_s * (i * k - j * r)
    m21 = two_s * (j * k + i * r)
    m22 = 1 - two_s * (i * i + j * j)

    # Stack to (..., 9) then reshape to (..., 3, 3)
    flat = mx.stack(
        [m00, m01, m02, m10, m11, m12, m20, m21, m22], axis=-1
    )
    return flat.reshape(*flat.shape[:-1], 3, 3)


def _diag_embed_3(x: mx.array) -> mx.array:
    """Equivalent of torch.diag_embed for last-dim-3 vectors.

    x shape (..., 3) -> output shape (..., 3, 3) with diagonal set to x.
    """
    shape = x.shape
    out = mx.zeros((*shape[:-1], 3, 3), dtype=x.dtype)
    # Build via stacking columns since mlx lacks a direct in-place diag set
    e0 = mx.stack([x[..., 0], mx.zeros_like(x[..., 0]), mx.zeros_like(x[..., 0])], axis=-1)
    e1 = mx.stack([mx.zeros_like(x[..., 1]), x[..., 1], mx.zeros_like(x[..., 1])], axis=-1)
    e2 = mx.stack([mx.zeros_like(x[..., 2]), mx.zeros_like(x[..., 2]), x[..., 2]], axis=-1)
    out = mx.stack([e0, e1, e2], axis=-2)  # (..., 3, 3)
    return out


def _build_covariance(scale: mx.array, rotation_xyzw: mx.array) -> mx.array:
    """Replica of upstream `gaussians.build_covariance`.

    Σ = R S Sᵀ Rᵀ  where S = diag(scale), R = quat_to_mat(rotation).
    """
    S = _diag_embed_3(scale)                       # (..., 3, 3)
    R = _quaternion_to_matrix(rotation_xyzw)        # (..., 3, 3)
    S_t = mx.swapaxes(S, -1, -2)
    R_t = mx.swapaxes(R, -1, -2)
    return R @ S @ S_t @ R_t


def _homogenize_points(p: mx.array) -> mx.array:
    """Append a `1` along the last axis."""
    ones = mx.ones((*p.shape[:-1], 1), dtype=p.dtype)
    return mx.concatenate([p, ones], axis=-1)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class GaussianAdapter(nn.Module):
    """MLX equivalent of upstream `UnifiedGaussianAdapter`.

    The "unified" variant is the one actually used by `EncoderYoNoSplat`
    (the base `GaussianAdapter.forward` requires extra projection terms
    and is unused at inference time).

    Construction takes the cfg + sprint metadata (num_surfaces,
    gaussians_per_axis, upscale_token_ratio); the latter three are stored
    for the caller's convenience but do not affect this module's math
    (the caller is responsible for reshaping the raw token output before
    invocation).
    """

    cfg: GaussianAdapterCfg
    sh_mask: mx.array

    def __init__(
        self,
        cfg: GaussianAdapterCfg,
        num_surfaces: int = 1,
        gaussians_per_axis: int = 14,
        upscale_token_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_surfaces = num_surfaces
        self.gaussians_per_axis = gaussians_per_axis
        self.upscale_token_ratio = upscale_token_ratio

        # SH mask: DC component = 1, higher-degree coefficients down-scaled.
        d_sh = self.d_sh
        mask = [1.0] * d_sh
        for degree in range(1, cfg.sh_degree + 1):
            lo = degree ** 2
            hi = (degree + 1) ** 2
            for idx in range(lo, hi):
                mask[idx] = 0.1 * (0.25 ** degree)
        self.sh_mask = mx.array(mask, dtype=mx.float32)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        """Per-Gaussian raw width (no opacity): 3 scale + 4 rot + 3*d_sh."""
        return 7 + 3 * self.d_sh

    @property
    def raw_gs_dim(self) -> int:
        """Per-Gaussian width *including* the opacity logit channel."""
        return 1 + self.d_in

    # ------------------------------------------------------------------
    # Forward (matches UnifiedGaussianAdapter signature)
    # ------------------------------------------------------------------

    def forward(
        self,
        means: mx.array,                    # (..., 3)         from point_head
        depths: mx.array,                   # (..., 1)         optional / informational
        opacities: mx.array,                # (..., 1) or (...) already pdf-mapped
        raw_gaussians: mx.array,            # (..., 10)        scale|rot|sh
        eps: float = 1e-8,
        extrinsics: Optional[mx.array] = None,   # (..., 4, 4)
        intrinsics: Optional[mx.array] = None,   # unused (kept for parity)
        coordinates: Optional[mx.array] = None,  # unused (kept for parity)
    ) -> Gaussians:
        d_sh = self.d_sh

        # Split the per-Gaussian raw vector.
        scales = raw_gaussians[..., 0:3]
        rotations = raw_gaussians[..., 3:7]
        sh = raw_gaussians[..., 7 : 7 + 3 * d_sh]

        # ----- scale activation (UnifiedGaussianAdapter) -----
        scales = 0.001 * _softplus(scales)
        scales = mx.minimum(scales, mx.array(0.3, dtype=scales.dtype))

        # ----- quaternion normalisation -----
        rot_norm = mx.sqrt((rotations * rotations).sum(axis=-1, keepdims=True))
        rotations = rotations / (rot_norm + eps)

        # ----- spherical harmonics -----
        # (..., 3*d_sh) -> (..., 3, d_sh)
        sh = sh.reshape(*sh.shape[:-1], 3, d_sh)
        # Upstream rule (verbatim):
        #     sh.broadcast_to((*opacities.shape, 3, d_sh))
        # The encoder calls us with opacities of shape (..., 1) and
        # raw_gaussians with an extra spp axis inserted, so the trailing
        # `1` axis on opacities is meaningful (it lines up with the spp
        # axis on sh) — we must NOT squeeze it.
        target_shape = (*tuple(opacities.shape), 3, d_sh)
        sh = mx.broadcast_to(sh, target_shape) * self.sh_mask

        # ----- covariance -----
        cov = _build_covariance(scales, rotations)

        if extrinsics is not None:
            c2w_rot = extrinsics[..., :3, :3]
            cov = c2w_rot @ cov @ mx.swapaxes(c2w_rot, -1, -2)

            # Transform means: (..., i, j) @ (..., j) -> (..., i)
            means_h = _homogenize_points(means)             # (..., 4)
            # einsum equivalent: out[..., i] = sum_j extrinsics[..., i, j] * means_h[..., j]
            means_h_exp = means_h[..., None, :]              # (..., 1, 4)
            means = (extrinsics * means_h_exp).sum(axis=-1)  # (..., 4)
            means = means[..., :3]

        # Broadcast rotations to scales' leading shape (so the per-spp axis
        # picks up). Upstream uses `rotations.broadcast_to((*scales.shape[:-1], 4))`.
        rotations_b = mx.broadcast_to(rotations, (*scales.shape[:-1], 4))

        return Gaussians(
            means=means,
            covariances=cov,
            scales=scales,
            rotations=rotations_b,
            harmonics=sh,
            opacities=opacities,
        )

    # Allow `adapter(...)` shorthand.
    def __call__(self, *args, **kwargs) -> Gaussians:
        return self.forward(*args, **kwargs)
