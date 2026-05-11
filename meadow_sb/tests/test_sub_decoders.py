"""Numerical parity test for the 3 sub-decoder MLX ports.

Asserts that for each of (point, gaussian, camera) decoders the MLX
implementation, when fed the exact same backbone-output tokens as PyTorch
upstream and loaded from the same checkpoint weights, produces an output
that differs from the PyTorch reference by less than `1e-3` (max abs).

Reference activations live in
`research/yonosplat_bootstrap/dumps/per_block/sub_<name>.npz` and are
produced by `research/yonosplat_bootstrap/scripts/dump_sub_decoders.py`.

If the reference dumps are missing (e.g. on CI without the heavy checkpoint),
the test is **skipped** rather than failed — these are integration-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mlx_core = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

import mlx.core as mx                          # noqa: E402

from meadow_sb.models.sub_decoders import (    # noqa: E402
    CameraDecoder,
    GaussianDecoder,
    PointDecoder,
    RoPE2D,
    load_sub_decoder_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMPS = REPO_ROOT / "research" / "yonosplat_bootstrap" / "dumps" / "per_block"
CKPT = (
    REPO_ROOT
    / "research"
    / "yonosplat_bootstrap"
    / "weights"
    / "yonosplat"
    / "re10k_224x224_ctx2to32.ckpt"
)


def _have_dumps() -> bool:
    return all(
        (DUMPS / f"sub_{n}.npz").exists() for n in ("point", "gaussian", "camera")
    )


def _have_ckpt() -> bool:
    return CKPT.exists()


@pytest.fixture(scope="module")
def pt_state_dict():
    """Load the upstream checkpoint state dict (CPU, no grad)."""
    if not _have_ckpt():
        pytest.skip(f"checkpoint not present at {CKPT}")
    import torch  # local import to keep the test importable without torch
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # Convert just the encoder.{point,gaussian,camera}_decoder.* slice to np.
    keep = ("encoder.point_decoder.", "encoder.gaussian_decoder.", "encoder.camera_decoder.")
    return {k: v.detach().cpu().numpy() for k, v in sd.items() if k.startswith(keep)}


@pytest.mark.skipif(not _have_dumps(), reason="sub-decoder dumps missing — run dump_sub_decoders.py")
@pytest.mark.parametrize(
    "name,decoder_cls,prefix,tol",
    [
        ("point",    PointDecoder,    "encoder.point_decoder.",    1e-3),
        ("gaussian", GaussianDecoder, "encoder.gaussian_decoder.", 1e-3),
        ("camera",   CameraDecoder,   "encoder.camera_decoder.",   1e-3),
    ],
)
def test_sub_decoder_matches_pt(pt_state_dict, name, decoder_cls, prefix, tol):
    ref = np.load(DUMPS / f"sub_{name}.npz")
    h_in_np = ref["in"]                          # (B*V, N, 2048) float32
    pos_np = ref["xpos"].astype(np.int32)        # (B*V, N, 2) int
    out_ref = ref["out"]                         # (B*V, N, out_dim)

    # Share one RoPE2D across the build — matches PT's `rope=self.backbone.rope`.
    rope = RoPE2D(freq=100.0)
    decoder = decoder_cls(rope=rope)
    load_sub_decoder_weights(decoder, pt_state_dict, prefix=prefix)
    decoder.eval()

    h_in_mx = mx.array(h_in_np)
    pos_mx = mx.array(pos_np)
    out_mx = decoder(h_in_mx, xpos=pos_mx)
    out_np = np.asarray(out_mx)

    assert out_np.shape == out_ref.shape, (
        f"[{name}] shape mismatch: MLX {out_np.shape} vs PT {out_ref.shape}"
    )

    abs_diff = np.abs(out_np - out_ref)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    out_scale = float(np.abs(out_ref).mean())
    print(
        f"[{name}] max|Δ|={max_diff:.3e} mean|Δ|={mean_diff:.3e} "
        f"|out|≈{out_scale:.3f}"
    )
    assert max_diff < tol, (
        f"[{name}] max abs diff {max_diff:.3e} >= tol {tol} "
        f"(mean {mean_diff:.3e}, ref scale {out_scale:.3f})"
    )
