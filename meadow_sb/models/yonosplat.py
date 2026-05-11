"""Top-level YoNoSplat encoder assembler — wires the 6 ported MLX modules.

Inference-only path; training-only knobs (scheduled GT-pose sampling, autocast,
loss heads, etc.) are intentionally omitted. The forward pass mirrors the
upstream `EncoderYoNoSplat.forward` flow but stays MLX-native end-to-end:

  image, intrinsics
    → DINOv2 encoder
    → (TODO) intrinsics_embed_layer  (skipped — module not yet ported)
    → CroCo decoder (returns hidden+pos with 5 register prefix tokens)
    → bilinear-upsample 2× the patch tokens
    → 3 sub-decoders in parallel: point / gaussian (+ rgb-embed add) / camera
    → 3 heads + pixel_shuffle (point + gaussian) / SO(3) + 4×4 c2w (camera)
    → GaussianAdapter packs everything into a `Gaussians` dataclass
    → return (gaussians, c2w_poses, predicted_intrinsics)

Five known gaps vs upstream are stubbed in-place and flagged with `STUB:`
comments — see docs/YONOSPLAT_INTEGRATION_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Sub-modules ported by the 8-agent parallel sprint.
from .dinov2_encoder import DinoVisionTransformer, load_encoder_from_state_dict
from .croco_decoder import (
    CroCoDecoder,
    build_decoder_positions,
    load_decoder_weights_from_pt,
)
from .sub_decoders import (
    PointDecoder,
    GaussianDecoder,
    CameraDecoder,
    load_sub_decoder_weights,
)
from .heads import (
    GaussianHead,
    PointHead,
    CameraHead,
    IntrinsicHead,
    RgbEmbed,
    load_gaussian_head,
    load_point_head,
    load_camera_head,
    load_intrinsic_head,
    load_rgb_embed,
)
from .gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, Gaussians


@dataclass
class YoNoSplatEncoderCfg:
    patch_size: int = 14
    embed_dim: int = 1024
    gaussians_per_axis: int = 14
    upscale_token_ratio: int = 2
    num_surfaces: int = 1
    sh_degree: int = 0
    num_register_tokens_enc: int = 4   # DINOv2 has 4 register tokens
    num_register_tokens_dec: int = 5   # CroCo decoder prepends 5
    gaussian_scale_min: float = 0.5
    gaussian_scale_max: float = 15.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def svd_orthogonalise(R: mx.array) -> mx.array:
    """Project a (..., 3, 3) matrix to the nearest SO(3) via SVD.

    MLX 0.31 lacks `mx.linalg.svd`. Round-trip through numpy at this boundary —
    the matrices are tiny (B*V × 3 × 3) and only run once per forward.
    """
    R_np = np.asarray(R, dtype=np.float32)
    U, _, Vt = np.linalg.svd(R_np)
    det = np.linalg.det(U @ Vt)
    D = np.broadcast_to(np.eye(3, dtype=np.float32), R_np.shape).copy()
    D[..., 2, 2] = det
    R_so3 = U @ D @ Vt
    return mx.array(R_so3)


def se3_inverse(T: mx.array) -> mx.array:
    """Invert a (..., 4, 4) rigid transform: R^T and -R^T t."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    R_inv = mx.swapaxes(R, -1, -2)
    t_inv = -(R_inv @ t)
    top = mx.concatenate([R_inv, t_inv], axis=-1)
    bottom = mx.broadcast_to(
        mx.array([0.0, 0.0, 0.0, 1.0], dtype=T.dtype).reshape(*([1] * (T.ndim - 2)), 1, 4),
        (*T.shape[:-2], 1, 4),
    )
    return mx.concatenate([top, bottom], axis=-2)


def _pixel_shuffle(x: mx.array, upscale: int) -> mx.array:
    """PyTorch-style pixel_shuffle.

    Input: (B, C, H, W) where C = c_out * upscale^2.
    Output: (B, c_out, H*upscale, W*upscale).
    """
    B, C, H, W = x.shape
    r = upscale
    c_out = C // (r * r)
    x = x.reshape(B, c_out, r, r, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3)        # (B, c_out, H, r, W, r)
    x = x.reshape(B, c_out, H * r, W * r)
    return x


def _bilinear_2x(x_bnc: mx.array, h: int, w: int) -> mx.array:
    """Bilinear 2× upsample of tokens given grid (h, w).

    Input:  (B, h*w, C)
    Output: (B, 2h*2w, C)
    """
    B, _, C = x_bnc.shape
    x_bhwc = x_bnc.reshape(B, h, w, C)
    up = nn.Upsample(scale_factor=2.0, mode="linear", align_corners=False)
    x_up = up(x_bhwc)                           # (B, 2h, 2w, C)
    return x_up.reshape(B, 2 * h * 2 * w, C)


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------

class YoNoSplatEncoder:
    """Top-level YoNoSplat encoder, MLX-native, inference-only.

    Build with `cfg` and a `state_dict` (PyTorch torch.load(...)['state_dict']).
    """

    def __init__(
        self,
        cfg: YoNoSplatEncoderCfg | None = None,
        state_dict: dict[str, Any] | None = None,
    ):
        self.cfg = cfg or YoNoSplatEncoderCfg()

        self.encoder: DinoVisionTransformer | None = None
        self.decoder: CroCoDecoder | None = None
        self.point_decoder: PointDecoder | None = None
        self.gaussian_decoder: GaussianDecoder | None = None
        self.camera_decoder: CameraDecoder | None = None
        self.point_head: PointHead | None = None
        self.gaussian_head: GaussianHead | None = None
        self.camera_head: CameraHead | None = None
        self.intrinsic_head: IntrinsicHead | None = None
        self.rgb_embed: RgbEmbed | None = None
        self.adapter: GaussianAdapter | None = None

        # Imagenet normalisation buffers.
        self.image_mean = mx.array([0.485, 0.456, 0.406], dtype=mx.float32).reshape(1, 3, 1, 1)
        self.image_std = mx.array([0.229, 0.224, 0.225], dtype=mx.float32).reshape(1, 3, 1, 1)

        if state_dict is not None:
            self.from_state_dict(state_dict)

    # ----------------------------------------------------------------------
    # Weight loading
    # ----------------------------------------------------------------------
    def from_state_dict(self, sd: dict) -> "YoNoSplatEncoder":
        cfg = self.cfg

        # DINOv2 encoder — instantiate then in-place load.
        self.encoder = DinoVisionTransformer(
            img_size=224,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=24,
            num_heads=16,
            num_register_tokens=cfg.num_register_tokens_enc,
        )
        load_encoder_from_state_dict(self.encoder, sd, prefix="encoder.backbone.encoder.")

        # STUB 4: DINOv2 pos_embed in re10k ckpt is (1, 1370, 1024) = 37x37
        # grid + cls; our forward sees 224×224 inputs → 16x16 = 256 patches.
        # Agent A's `interpolate_pos_encoding()` raises NotImplementedError.
        # Bicubic interp here once at load time so the forward sees a
        # matched (1, 257, 1024) tensor.
        pe = self.encoder.pos_embed  # (1, 1370, 1024)
        n_old = pe.shape[1] - 1  # 1369
        old_grid = int(round(n_old ** 0.5))  # 37
        new_grid = 224 // cfg.patch_size  # 16
        if old_grid != new_grid:
            import torch as _torch
            import torch.nn.functional as _F
            cls = pe[:, :1, :]
            patches = pe[:, 1:, :]  # (1, old², C)
            patches_t = _torch.from_numpy(np.asarray(patches)).reshape(
                1, old_grid, old_grid, cfg.embed_dim
            ).permute(0, 3, 1, 2)  # (1, C, og, og)
            patches_up = _F.interpolate(
                patches_t.float(),
                size=(new_grid, new_grid),
                mode="bicubic",
                align_corners=False,
            )
            patches_up = patches_up.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, cfg.embed_dim)
            new_pe = mx.concatenate([cls, mx.array(patches_up.numpy())], axis=1)
            self.encoder.pos_embed = new_pe

        # CroCo decoder.
        self.decoder = CroCoDecoder(
            dim=cfg.embed_dim,
            num_heads=16,
            depth=36,
            num_register_tokens=cfg.num_register_tokens_dec,
        )
        load_decoder_weights_from_pt(self.decoder, sd)

        # Sub-decoders.
        self.point_decoder = PointDecoder()
        load_sub_decoder_weights(self.point_decoder, sd, prefix="encoder.point_decoder.")
        self.gaussian_decoder = GaussianDecoder()
        load_sub_decoder_weights(self.gaussian_decoder, sd, prefix="encoder.gaussian_decoder.")
        self.camera_decoder = CameraDecoder()
        load_sub_decoder_weights(self.camera_decoder, sd, prefix="encoder.camera_decoder.")

        # Heads (factory functions return loaded instances). NOTE: head
        # loaders expect prefix WITHOUT trailing dot (they append `.proj.x`
        # themselves), unlike the encoder/sub_decoder loaders.
        self.point_head = load_point_head(sd, prefix="encoder.point_head")
        self.gaussian_head = load_gaussian_head(sd, prefix="encoder.gaussian_head")
        self.camera_head = load_camera_head(sd, prefix="encoder.camera_head")
        self.intrinsic_head = load_intrinsic_head(sd, prefix="encoder.backbone.intrinsic_head")
        self.rgb_embed = load_rgb_embed(sd, prefix="encoder.rgb_embed")

        # Adapter (no weights — pure arithmetic).
        self.adapter = GaussianAdapter(
            GaussianAdapterCfg(
                sh_degree=cfg.sh_degree,
                gaussian_scale_min=cfg.gaussian_scale_min,
                gaussian_scale_max=cfg.gaussian_scale_max,
            ),
            num_surfaces=cfg.num_surfaces,
            gaussians_per_axis=cfg.gaussians_per_axis,
            upscale_token_ratio=cfg.upscale_token_ratio,
        )
        return self

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def __call__(
        self,
        images: mx.array,                  # (B, V, 3, H, W) in [0, 1]
        intrinsics: mx.array | None = None,  # (B, V, 3, 3) — currently unused (see stub 2)
    ) -> dict:
        assert self.encoder is not None, "call from_state_dict() first"
        cfg = self.cfg
        ps = cfg.patch_size
        ust = cfg.upscale_token_ratio

        B, V, C, H, W = images.shape
        h, w = H // ps, W // ps
        BV = B * V

        # ---- 1. Normalise + flatten views into batch ----
        imgs = (images - self.image_mean[None]) / self.image_std[None]  # (B, V, 3, H, W)
        imgs = imgs.reshape(BV, C, H, W)
        imgs_bhwc = mx.transpose(imgs, (0, 2, 3, 1))                    # MLX channels-last

        # ---- 2. DINOv2 encoder ----
        enc_out = self.encoder(imgs_bhwc)
        patch_tokens = enc_out["x_norm_patchtokens"]                    # (BV, h*w, 1024)
        cls_token = enc_out["x_norm_clstoken"]                          # (BV, 1024)

        # ---- 3. Intrinsic prediction (fx, fy) ----
        intrinsic_pred = nn.relu(self.intrinsic_head(cls_token))        # (BV, 2)

        # STUB 1: intrinsics_embed_layer not ported. Upstream would do
        #   hidden = patch_tokens + intrinsics_embed_layer(harmonic_embed(intrinsics, imgs))
        # The layer is zero-init at construction; in re10k it is trained but
        # contributes a small additive bias. Skipping it costs numerical
        # parity but lets the forward run end-to-end.
        hidden = patch_tokens

        # ---- 4. CroCo decoder ----
        dec_hidden, dec_pos = self.decoder(hidden, B, V, H, W, ps)      # (BV, 5+h*w, 1024)
        psi = cfg.num_register_tokens_dec   # patch_start_idx = 5

        # STUB 2: Upstream concats the last-two decoder block outputs (dim
        # doubles to 2048). Agent B's port returns only the final block. As a
        # placeholder we tile dim → 2*dim so the downstream sub-decoders
        # (in_dim=2048) accept the shape. Numerical parity requires patching
        # the decoder to capture both layers.
        dec_hidden2x = mx.concatenate([dec_hidden, dec_hidden], axis=-1)  # (BV, 5+h*w, 2048)

        # ---- 5. Upsample 2× the patch portion of the decoder output ----
        aux = dec_hidden2x[:, :psi, :]                                   # (BV, 5, 2048)
        patch = dec_hidden2x[:, psi:, :]                                 # (BV, h*w, 2048)
        patch_up = _bilinear_2x(patch, h, w)                             # (BV, 2h*2w, 2048)
        hidden_upsampled = mx.concatenate([aux, patch_up], axis=1)       # (BV, 5+4hw, 2048)

        # Upsampled positions.
        pos_aux = dec_pos[:, :psi]
        pos_img_up = build_decoder_positions(B, V, h * ust, w * ust, 0)  # no register prefix
        pos_img_up = pos_img_up + 1                                      # +1 for special-token offset
        pos_upsampled = mx.concatenate([pos_aux, pos_img_up], axis=1)

        # ---- 6. RGB embed → add to gaussian path ----
        rgb_bhwc = mx.transpose(imgs.reshape(BV, C, H, W) * self.image_std.reshape(1, C, 1, 1)
                                + self.image_mean.reshape(1, C, 1, 1), (0, 2, 3, 1))
        # Actually upstream feeds the *normalised-back* RGB? No — it feeds raw
        # images. images is (B, V, C, H, W) in [0,1]; flatten and pass through.
        rgb_in = mx.transpose(images.reshape(BV, C, H, W), (0, 2, 3, 1))  # (BV, H, W, C)
        rgb_feat = self.rgb_embed(rgb_in)                                # (BV, 1024, 2048) for 224/7=32

        # The rgb_feat shape is (BV, (H/7)*(W/7), 2048) = (BV, 4hw, 2048) when
        # H/W = 224 and upscale_token_ratio=2. Add to the upsampled patch
        # portion of the gaussian path.
        gauss_patch = patch_up + rgb_feat                                # (BV, 4hw, 2048)
        hidden_gaussian = mx.concatenate([aux, gauss_patch], axis=1)     # (BV, 5+4hw, 2048)

        # ---- 7. Three sub-decoders ----
        point_hidden = self.point_decoder(hidden_upsampled, xpos=pos_upsampled)    # (BV, 5+4hw, 1024)
        gaussian_hidden = self.gaussian_decoder(hidden_gaussian, xpos=pos_upsampled)
        camera_hidden = self.camera_decoder(dec_hidden2x, xpos=dec_pos)            # (BV, 5+h*w, 512)

        # ---- 8. Heads + pixel_shuffle ----
        out_h, out_w = h * cfg.gaussians_per_axis, w * cfg.gaussians_per_axis
        upsampled_h, upsampled_w = h * ust, w * ust  # 32, 32 for 224 inputs

        # 8a. Point head: 1024 → 147 (= 3 × 7²). pixel_shuffle to (B*V, 3, out_h, out_w).
        # STUB 3: heads.py PointHead is bare Linear (147-d). pixel_shuffle is
        # assembler responsibility (mirrors upstream LinearPts3d.forward).
        pt_feat = self.point_head(point_hidden[:, psi:])                          # (BV, 4hw, 147)
        # reshape (BV, 4hw, 147) → (BV, 147, 2h, 2w) → pixel_shuffle 7 → (BV, 3, 14h, 14w)
        pt_bchw = pt_feat.transpose(0, 2, 1).reshape(BV, 147, upsampled_h, upsampled_w)
        pt_shuf = _pixel_shuffle(pt_bchw, upscale=ps // ust)                     # (BV, 3, out_h, out_w)
        # back to (B, V, out_h, out_w, 3) — split into xy, z; z = exp(z); concat
        pt_bvhwc = pt_shuf.reshape(B, V, 3, out_h, out_w).transpose(0, 1, 3, 4, 2)
        xy = pt_bvhwc[..., :2]
        z = mx.exp(pt_bvhwc[..., 2:3])
        local_points = mx.concatenate([xy * z, z], axis=-1)                       # (B, V, out_h, out_w, 3)

        # 8b. Gaussian head: 1024 → 539 (= 11 × 7²).
        gs_feat = self.gaussian_head(gaussian_hidden[:, psi:])                    # (BV, 4hw, 539)
        gs_bchw = gs_feat.transpose(0, 2, 1).reshape(BV, 539, upsampled_h, upsampled_w)
        gs_shuf = _pixel_shuffle(gs_bchw, upscale=ps // ust)                     # (BV, 11, out_h, out_w)
        gs_bvhwd = gs_shuf.reshape(B, V, 11, out_h, out_w).transpose(0, 1, 3, 4, 2)

        # 8c. Camera head: (BV, h*w, 512) → (BV, 12).
        cam_raw = self.camera_head(camera_hidden[:, psi:], h, w)                 # (BV, 12)
        rot9 = cam_raw[:, :9].reshape(BV, 3, 3)
        t3 = cam_raw[:, 9:12]
        rot_so3 = svd_orthogonalise(rot9)                                        # (BV, 3, 3)
        # Build 4×4 c2w.
        top = mx.concatenate([rot_so3, t3.reshape(BV, 3, 1)], axis=-1)           # (BV, 3, 4)
        last_row = mx.broadcast_to(
            mx.array([0.0, 0.0, 0.0, 1.0], dtype=top.dtype).reshape(1, 1, 4),
            (BV, 1, 4),
        )
        camera_poses = mx.concatenate([top, last_row], axis=-2).reshape(B, V, 4, 4)

        # Convert to first-view-centric frame.
        w2c_v1 = se3_inverse(camera_poses[:, 0])                                 # (B, 4, 4)
        # camera_poses_new[:, i] = w2c_v1 @ camera_poses[:, i]
        cp_flat = camera_poses.reshape(B, V, 4, 4)
        camera_poses = mx.einsum("bij,bnjk->bnik", w2c_v1, cp_flat)              # (B, V, 4, 4)

        # ---- 9. GaussianAdapter ----
        # Pack as upstream does.
        pts_all = local_points.reshape(B, V, out_h * out_w, 3)
        # add num_surfaces and spp axes: (B, V, L, S, 1, 3)
        pts_means = pts_all.reshape(B, V, -1, 1, 1, 3)
        depths = pts_means[..., -1:]
        gs_flat = gs_bvhwd.reshape(B, V, out_h * out_w, 11)
        # split into (srf, c) = (1, 11)
        gs_split = gs_flat.reshape(B, V, -1, cfg.num_surfaces, 11)
        opacities = mx.sigmoid(gs_split[..., 0:1])                               # (B, V, L, S, 1)
        raw_gaussians = gs_split[..., 1:].reshape(B, V, -1, cfg.num_surfaces, 1, 10)

        extrinsics = camera_poses.reshape(B, V, 1, 1, 1, 4, 4)
        gaussians = self.adapter.forward(
            means=pts_means,
            depths=depths,
            opacities=opacities,
            raw_gaussians=raw_gaussians,
            extrinsics=extrinsics,
        )

        return {
            "gaussians": gaussians,
            "camera_poses": camera_poses,
            "intrinsic_pred": intrinsic_pred.reshape(B, V, 2),
            "local_points": local_points,
        }
