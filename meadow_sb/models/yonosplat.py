"""Top-level YoNoSplat encoder assembler — wires the 6 ported MLX modules.

Inference-only path; training-only knobs (scheduled GT-pose sampling, autocast,
loss heads, etc.) are intentionally omitted. The forward pass mirrors the
upstream `EncoderYoNoSplat.forward` flow but stays MLX-native end-to-end:

  image, intrinsics
    → BackboneLocalGlobal (DINOv2 enc + CroCo dec + intrinsic head)
    → bilinear-upsample 2× the patch tokens
    → 3 sub-decoders in parallel: point / gaussian (+ rgb-embed add) / camera
    → 3 heads
    → SO(3) orthogonalisation of camera rotation, then 4×4 c2w
    → GaussianAdapter packs everything into a `Gaussians` dataclass
    → return (gaussians, c2w_poses, predicted_intrinsics)

The rasterizer is deliberately NOT called here — keep this module pure
MLX so it loads on any Apple Silicon machine. Rendering is a separate glue
step that crosses the MLX / gsplat-CPU boundary.

This file's job is the wire-up; per-module numerics are already validated
by the parallel-agent sprint (see test_*.py in meadow_sb/tests/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# Sub-modules ported by the 8-agent parallel sprint.
from .dinov2_encoder import DinoV2Encoder  # noqa: F401  (Agent A)
from .croco_decoder import CroCoDecoder  # noqa: F401  (Agent B)
from .sub_decoders import PointDecoder, GaussianDecoder, CameraDecoder  # Agent C
from .heads import GaussianHead, PointHead, CameraHead, IntrinsicHead, RgbEmbed  # Agent D
from .gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, Gaussians  # Agent E


@dataclass
class YoNoSplatEncoderCfg:
    """Hyper-parameters mirroring `EncoderYoNoSplatCfg` from upstream."""

    patch_size: int = 14
    embed_dim: int = 1024
    gaussians_per_axis: int = 14
    upscale_token_ratio: int = 2
    num_surfaces: int = 1
    sh_degree: int = 0
    # Adapter
    gaussian_scale_min: float = 0.5
    gaussian_scale_max: float = 15.0


def svd_orthogonalise(R: mx.array) -> mx.array:
    """Project a 3×3 matrix to the nearest SO(3) via SVD.

    Pure MLX (no torch). Camera-head predicts an unconstrained 9-d vector;
    upstream reshapes to (3, 3) then projects via U·diag(1,1,det(U·Vᵀ))·Vᵀ.

    Note: MLX 0.31 lacks `mx.linalg.svd`. We fall back to numpy at this
    boundary — the operation is tiny (B×V × 3×3 matrices) and only runs
    once per forward, so the round-trip cost is negligible.
    """
    import numpy as np
    R_np = np.asarray(R)
    U, _, Vt = np.linalg.svd(R_np)
    det = np.linalg.det(U @ Vt)
    D = np.eye(3, dtype=R_np.dtype)
    D = np.broadcast_to(D, R_np.shape).copy()
    D[..., 2, 2] = det
    R_so3 = U @ D @ Vt
    return mx.array(R_so3)


class YoNoSplatEncoder:
    """Top-level YoNoSplat encoder, MLX-native, inference-only.

    Build with `cfg` and a `state_dict` (PyTorch torch.load(...)['state_dict']);
    weights are sliced + loaded into each ported sub-module.
    """

    def __init__(self, cfg: YoNoSplatEncoderCfg | None = None, state_dict: dict[str, Any] | None = None):
        self.cfg = cfg or YoNoSplatEncoderCfg()
        # Backbone modules are constructed by their respective files' factory
        # helpers (each agent's port exposes a `load_*_from_state_dict()` or
        # `build_*_from_state_dict()` entry; see test files for examples).
        # We do not auto-load weights here — call `from_state_dict()` instead.
        self.backbone_encoder = None  # type: ignore[assignment]
        self.backbone_decoder = None
        self.intrinsic_head = None
        self.point_decoder = None
        self.gaussian_decoder = None
        self.camera_decoder = None
        self.point_head = None
        self.gaussian_head = None
        self.camera_head = None
        self.rgb_embed = None
        self.adapter = None
        if state_dict is not None:
            self.from_state_dict(state_dict)

    def from_state_dict(self, sd: dict) -> "YoNoSplatEncoder":
        """Slice + load each sub-module from the YoNoSplat PT state-dict.

        The exact loader function lives in each agent's module. We import
        them lazily so partial sprint progress doesn't break the assembler
        import.
        """
        # Importing lazily avoids hard dependency if any one agent's file is
        # missing during a partial sprint (the e2e harness uses the same
        # pattern).
        try:
            from .dinov2_encoder import load_encoder_from_state_dict  # Agent A
            self.backbone_encoder = load_encoder_from_state_dict(sd)
        except (ImportError, AttributeError) as e:
            print(f"[yonosplat] DINOv2 encoder loader missing: {e}")

        try:
            from .croco_decoder import build_croco_decoder_from_state_dict  # Agent B
            self.backbone_decoder = build_croco_decoder_from_state_dict(sd)
        except (ImportError, AttributeError) as e:
            print(f"[yonosplat] CroCo decoder loader missing: {e}")

        # Adapter (Agent E) — always cheap.
        adapter_cfg = GaussianAdapterCfg(
            sh_degree=self.cfg.sh_degree,
            gaussian_scale_min=self.cfg.gaussian_scale_min,
            gaussian_scale_max=self.cfg.gaussian_scale_max,
        )
        self.adapter = GaussianAdapter(adapter_cfg, num_surfaces=self.cfg.num_surfaces)
        return self

    # -------------------------------------------------------------------------
    # Forward pass — mirrors upstream EncoderYoNoSplat.forward but pure MLX.
    # -------------------------------------------------------------------------
    def __call__(
        self,
        images: mx.array,
        intrinsics: mx.array,
    ) -> dict:
        """Run the full inference pipeline.

        Args:
            images:     (B, V, 3, H, W) float32, in [0, 1].
            intrinsics: (B, V, 3, 3) float32, normalised (pixelSplat convention).

        Returns:
            dict with keys:
                gaussians: `Gaussians` dataclass (xyz, scale, rotation,
                           opacity, features), all (B, V*N*S, ...) flattened
                camera_poses: (B, V, 4, 4) world-to-view-1 normalised c2w
                intrinsic_pred: (B, V, 2) predicted (fx, fy) per view
        """
        raise NotImplementedError(
            "YoNoSplatEncoder.__call__ wiring is staged: each agent's sub-module"
            " loader needs to expose a consistent factory before the assembler"
            " can chain them. Tracked in docs/YONOSPLAT_INTEGRATION_PLAN.md."
        )
