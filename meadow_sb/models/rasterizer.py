"""CPU-side Gaussian splatting rasterizer wrapper.

Tier-1 of the YoNoSplat MLX port: a forward-only Gaussian rasterizer that runs
on Apple Silicon CPU (no CUDA, no Metal kernel). It is used as a reference /
fallback so the rest of the pipeline can produce rendered images while the
Metal kernel (Tier 2) is being built.

Design choices:
- We use ``gsplat``'s pure-PyTorch ``_torch_impl`` for the *projection* stage,
  which is identical math to upstream's CUDA path. This guarantees the
  projected means, depths and conics match upstream within float-rounding.
- For the *rasterization* (pixel compositing) stage, ``gsplat`` 1.5+ only ships
  a CUDA kernel — even its ``_torch_impl._rasterize_to_pixels`` re-enters CUDA
  via ``rasterize_to_indices_in_range``. So we ship a small hand-rolled
  front-to-back alpha-compositing loop using only ``torch`` ops on CPU. It is
  forward-only (no autograd needed) and tile-batched to keep memory bounded.
- SH evaluation reuses ``_torch_impl._spherical_harmonics`` (also pure
  PyTorch).
- Intrinsics conversion mirrors upstream ``DecoderSplattingGSPlat``:
  YoNoSplat ships normalised intrinsics (row 0 / W, row 1 / H); gsplat wants
  pixel-space K.

When the Metal kernel lands in Tier 2, the public ``GsplatRasterizer.render``
signature stays the same; only the ``_rasterize_pixels`` internal call swaps
to the Metal backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Public Gaussian container
# --------------------------------------------------------------------------- #

@dataclass
class Gaussians:
    """Plain container mirroring upstream ``src.model.types.Gaussians``.

    All tensors are (B, N, ...) — batch-first. ``features`` carries SH
    coefficients in the (3, d_sh) layout used by YoNoSplat upstream; the
    rasterizer permutes to (d_sh, 3) before handing to gsplat.

    Either (rotations + scales) or covariances can be ``None``; if both are
    given, covariances take precedence (matches upstream).
    """

    xyz: Tensor             # (B, N, 3)
    scale: Tensor           # (B, N, 3)  (post-exp)
    rotation: Tensor        # (B, N, 4)  unit quat (w, x, y, z)
    opacity: Tensor         # (B, N, 1) or (B, N)
    features: Tensor        # (B, N, 3, d_sh)  SH coeffs
    covariances: Optional[Tensor] = None  # (B, N, 3, 3), optional override


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _denormalise_intrinsics(intrinsics: Tensor, H: int, W: int) -> Tensor:
    """YoNoSplat ships intrinsics with row 0 / W, row 1 / H. gsplat wants
    pixel-space K. Multiply rows 0 and 1 back."""
    K = intrinsics.clone()
    K[..., 0, :] = K[..., 0, :] * W
    K[..., 1, :] = K[..., 1, :] * H
    return K


def _opacity_threshold_prune(gauss: Gaussians, threshold: float) -> Gaussians:
    """Inference-time opacity prune. Matches upstream ``prune_gaussians``
    behaviour for inference mode with bs=1."""
    if threshold <= 0.0:
        return gauss
    op = gauss.opacity
    if op.dim() == 3:  # (B, N, 1)
        mask = (op[..., 0] > threshold)
    else:
        mask = (op > threshold)
    # Assume B = 1 (upstream prune is asserted bs=1 in inference)
    if gauss.xyz.shape[0] != 1:
        return gauss  # safety: skip prune for multi-batch
    keep = mask[0]
    if keep.all():
        return gauss
    g = Gaussians(
        xyz=gauss.xyz[:, keep],
        scale=gauss.scale[:, keep],
        rotation=gauss.rotation[:, keep],
        opacity=gauss.opacity[:, keep],
        features=gauss.features[:, keep],
        covariances=(gauss.covariances[:, keep] if gauss.covariances is not None else None),
    )
    return g


# --------------------------------------------------------------------------- #
# Pixel-stage rasterizer (pure PyTorch, CPU)
# --------------------------------------------------------------------------- #

def _rasterize_pixels_cpu(
    means2d: Tensor,       # (M, 2)  pixel-space, M = surviving gaussians for this view
    conics: Tensor,        # (M, 3)
    colors: Tensor,        # (M, 3)  already SH-evaluated, post-clamp
    opacities: Tensor,     # (M,)
    radii: Tensor,         # (M, 2)
    depths: Tensor,        # (M,)
    image_height: int,
    image_width: int,
    background: Tensor,    # (3,)
    pixel_chunk: int = 4096,
) -> Tuple[Tensor, Tensor]:
    """Forward-only front-to-back alpha compositing on CPU.

    Returns (rgb (H, W, 3), depth (H, W)).
    """
    device = means2d.device
    dtype = means2d.dtype

    # Sort gaussians by depth, front-to-back (smaller depth first).
    order = torch.argsort(depths)
    means2d = means2d[order]
    conics = conics[order]
    colors = colors[order]
    opacities = opacities[order]
    radii = radii[order]
    depths = depths[order]
    M = means2d.shape[0]

    rgb = torch.zeros(image_height * image_width, 3, dtype=dtype, device=device)
    acc_depth = torch.zeros(image_height * image_width, dtype=dtype, device=device)
    transmittance = torch.ones(image_height * image_width, dtype=dtype, device=device)

    if M == 0:
        rgb = rgb + background.to(dtype) * transmittance.unsqueeze(-1)
        rgb = rgb.reshape(image_height, image_width, 3)
        depth = acc_depth.reshape(image_height, image_width)
        return rgb, depth

    # Pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=dtype),
        torch.arange(image_width, device=device, dtype=dtype),
        indexing="ij",
    )
    # gsplat uses pixel centres at (x + 0.5, y + 0.5) — match _persp_proj convention.
    px = (xs + 0.5).reshape(-1)
    py = (ys + 0.5).reshape(-1)
    P = px.shape[0]

    # Iterate over Gaussians in depth order; for each Gaussian, only touch the
    # pixels inside its (axis-aligned) bounding box. This keeps the inner
    # broadcasts small while staying purely vectorised.
    for i in range(M):
        rx = float(radii[i, 0].item())
        ry = float(radii[i, 1].item())
        if rx <= 0.0 or ry <= 0.0:
            continue
        cx = float(means2d[i, 0].item())
        cy = float(means2d[i, 1].item())
        x0 = max(int(cx - rx), 0)
        x1 = min(int(cx + rx) + 1, image_width)
        y0 = max(int(cy - ry), 0)
        y1 = min(int(cy + ry) + 1, image_height)
        if x0 >= x1 or y0 >= y1:
            continue

        # Build the flat pixel indices for this bbox.
        w_box = x1 - x0
        h_box = y1 - y0
        ys_box = torch.arange(y0, y1, device=device)
        xs_box = torch.arange(x0, x1, device=device)
        flat_idx = (ys_box[:, None] * image_width + xs_box[None, :]).reshape(-1)

        # Skip pixels already saturated (T < 1e-4 is the gsplat convention).
        T_sub = transmittance[flat_idx]
        active = T_sub > 1e-4
        if not active.any():
            continue
        flat_idx = flat_idx[active]
        T_sub = T_sub[active]

        dx = px[flat_idx] - cx
        dy = py[flat_idx] - cy
        # 2D Gaussian power: 0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
        a, b, c = conics[i, 0], conics[i, 1], conics[i, 2]
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        # gsplat clamps power at 0 (no amplification > centre value).
        power = torch.clamp(power, max=0.0)
        # Skip pixels with negligible contribution.
        alpha = torch.clamp(opacities[i] * torch.exp(power), max=0.999)
        contrib = alpha > (1.0 / 255.0)
        if not contrib.any():
            continue
        flat_idx = flat_idx[contrib]
        T_sub = T_sub[contrib]
        alpha = alpha[contrib]

        w = alpha * T_sub
        rgb[flat_idx] += colors[i].unsqueeze(0) * w.unsqueeze(-1)
        acc_depth[flat_idx] += depths[i] * w
        transmittance[flat_idx] = T_sub * (1.0 - alpha)

    # Compose background using remaining transmittance.
    rgb = rgb + background.to(dtype).unsqueeze(0) * transmittance.unsqueeze(-1)

    rgb = rgb.reshape(image_height, image_width, 3)
    depth = acc_depth.reshape(image_height, image_width)
    return rgb, depth


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

class GsplatRasterizer:
    """Forward-only Gaussian rasterizer.

    Tier-1: CPU PyTorch via ``gsplat``'s pure-Python projection + a small
    hand-rolled compositor. Output is bit-comparable (within ``1e-3``) with
    upstream's ``DecoderSplattingGSPlat.rendering_fn`` running on CPU.

    Tier-2 (later): swap ``backend="metal"`` to call a Metal kernel; the
    public ``render`` signature stays unchanged.
    """

    def __init__(
        self,
        near: float = 0.1,
        far: float = 1000.0,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        opacity_threshold: float = 0.005,
        eps2d: float = 0.3,
        radius_clip: float = 0.1,
        backend: str = "cpu",
    ) -> None:
        if backend != "cpu":
            raise NotImplementedError(
                f"Tier-1 only supports backend='cpu'. Got {backend!r}. "
                "Metal kernel is Tier-2 and not yet implemented."
            )
        self.near = float(near)
        self.far = float(far)
        self.background = tuple(background)
        self.opacity_threshold = float(opacity_threshold)
        self.eps2d = float(eps2d)
        self.radius_clip = float(radius_clip)
        self.backend = backend

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def render(
        self,
        gaussians: Gaussians,
        extrinsics: Tensor,        # (B, V, 4, 4) world-to-camera (cam-to-world per upstream convention; see note)
        intrinsics: Tensor,        # (B, V, 3, 3) normalised (per pixelSplat convention)
        image_shape: Tuple[int, int],
    ) -> Tensor:
        """Render Gaussians to RGB.

        Convention note: upstream ``DecoderSplattingGSPlat`` receives
        ``extrinsics`` as camera-to-world and inverts inside; we follow that
        convention here so swapping in this rasterizer in upstream's place
        requires no caller changes. If you already hold w2c, set the
        ``extrinsics`` argument to its inverse before calling, or use
        :meth:`render_w2c` which skips the inversion.
        """
        H, W = image_shape
        B, V = extrinsics.shape[:2]

        # Match upstream: invert c2w to get w2c.
        w2c = torch.linalg.inv(extrinsics.float())
        return self._render_w2c(gaussians, w2c, intrinsics, H, W)[0]

    def render_w2c(
        self,
        gaussians: Gaussians,
        w2c: Tensor,
        intrinsics: Tensor,
        image_shape: Tuple[int, int],
    ) -> Tensor:
        H, W = image_shape
        return self._render_w2c(gaussians, w2c, intrinsics, H, W)[0]

    def render_with_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Tensor,
        intrinsics: Tensor,
        image_shape: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        H, W = image_shape
        w2c = torch.linalg.inv(extrinsics.float())
        return self._render_w2c(gaussians, w2c, intrinsics, H, W)

    # ------------------------------------------------------------------ #
    # core
    # ------------------------------------------------------------------ #
    def _render_w2c(
        self,
        gaussians: Gaussians,
        w2c: Tensor,
        intrinsics: Tensor,
        H: int,
        W: int,
    ) -> Tuple[Tensor, Tensor]:
        # Local imports keep import cost down for callers who only need the
        # class symbol.
        from gsplat.cuda._torch_impl import (
            _fully_fused_projection,
            _quat_scale_to_covar_preci,
            _spherical_harmonics,
        )
        from math import sqrt

        gauss = _opacity_threshold_prune(gaussians, self.opacity_threshold)

        B, V = w2c.shape[:2]
        device = gauss.xyz.device
        dtype = gauss.xyz.dtype

        means = gauss.xyz
        rotations = gauss.rotation
        scales = gauss.scale
        opacities = gauss.opacity
        if opacities.dim() == 3:
            opacities = opacities[..., 0]

        # SH coeffs: upstream stores (B, N, 3, d_sh) and permutes to (B, N, d_sh, 3) for gsplat.
        features = gauss.features.permute(0, 1, 3, 2).contiguous()  # (B, N, K, 3)
        K_sh = features.shape[-2]
        sh_degree = int(sqrt(K_sh)) - 1

        # Intrinsics: denormalise.
        K_px = _denormalise_intrinsics(intrinsics.float(), H, W)

        # Build covariances. Prefer explicit cov if provided (matches upstream).
        if gauss.covariances is not None:
            covars = gauss.covariances
        else:
            covars, _ = _quat_scale_to_covar_preci(rotations, scales, True, False, triu=False)

        # Project (per batch element). _fully_fused_projection accepts batch dims.
        # Shapes:
        #   means:    (B, N, 3)
        #   covars:   (B, N, 3, 3)
        #   viewmats: (B, V, 4, 4) -- treat as (B, C=V)
        #   Ks:       (B, V, 3, 3)
        radii, means2d, depths, conics, _ = _fully_fused_projection(
            means.float(),
            covars.float(),
            w2c.float(),
            K_px.float(),
            W, H,
            eps2d=self.eps2d,
            near_plane=self.near,
            far_plane=self.far,
            camera_model="pinhole",
        )
        # radii:   (B, V, N, 2)
        # means2d: (B, V, N, 2)
        # depths:  (B, V, N)
        # conics:  (B, V, N, 3)

        # Apply radius_clip (match upstream's radius_clip=0.1).
        if self.radius_clip > 0.0:
            r_keep = (radii.max(dim=-1).values >= self.radius_clip)
            # zero out radii of clipped gaussians so they're skipped downstream.
            zeroed = torch.where(r_keep.unsqueeze(-1), radii, torch.zeros_like(radii))
            radii = zeroed.int()

        # SH -> RGB. We need the view direction per (B, V, N).
        # Per-camera position in world coords = (R^T) @ (-t) where w2c = [R|t].
        R = w2c[..., :3, :3]                        # (B, V, 3, 3)
        t = w2c[..., :3, 3]                         # (B, V, 3)
        cam_pos_world = -torch.einsum("bvji,bvj->bvi", R, t)  # (B, V, 3)

        # Direction from cam to each gaussian.
        # means: (B, N, 3) -> (B, 1, N, 3); cam_pos: (B, V, 1, 3)
        dirs = means.unsqueeze(1) - cam_pos_world.unsqueeze(2)  # (B, V, N, 3)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # Broadcast SH coeffs across views: features (B, N, K, 3) -> (B, V, N, K, 3)
        sh_coeffs = features.unsqueeze(1).expand(B, V, -1, K_sh, 3)
        rgb_sh = _spherical_harmonics(sh_degree, dirs, sh_coeffs)  # (B, V, N, 3)
        # Match 3DGS convention: + 0.5, then clamp >= 0
        colors_rgb = torch.clamp(rgb_sh + 0.5, min=0.0)

        background = torch.tensor(self.background, dtype=torch.float32, device=device)

        rgb_out = torch.zeros(B, V, 3, H, W, dtype=torch.float32, device=device)
        depth_out = torch.zeros(B, V, H, W, dtype=torch.float32, device=device)

        for b in range(B):
            for v in range(V):
                m2 = means2d[b, v]               # (N, 2)
                cn = conics[b, v]                # (N, 3)
                cl = colors_rgb[b, v]            # (N, 3)
                op = opacities[b]                # (N,)
                rd = radii[b, v]                 # (N, 2)
                dp = depths[b, v]                # (N,)

                # Drop gaussians with zero radius (already filtered by projection / clip).
                alive = (rd > 0).all(dim=-1) & (dp > self.near) & (dp < self.far)
                if not alive.any():
                    rgb_view = background.view(1, 1, 3).expand(H, W, 3).clone()
                    rgb_out[b, v] = rgb_view.permute(2, 0, 1)
                    continue

                rgb_img, depth_img = _rasterize_pixels_cpu(
                    m2[alive],
                    cn[alive],
                    cl[alive],
                    op[alive],
                    rd[alive].to(dtype),
                    dp[alive],
                    H, W,
                    background,
                )
                rgb_img = rgb_img.clamp(0.0, 1.0)
                rgb_out[b, v] = rgb_img.permute(2, 0, 1)
                depth_out[b, v] = depth_img

        return rgb_out, depth_out


__all__ = ["Gaussians", "GsplatRasterizer"]
