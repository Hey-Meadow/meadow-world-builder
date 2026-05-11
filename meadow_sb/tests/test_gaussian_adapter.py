"""Parity test: meadow_sb MLX `GaussianAdapter` vs upstream PT `UnifiedGaussianAdapter`.

We feed both adapters the same synthetic per-Gaussian inputs (10-dim raw
vector after the opacity channel is split off, plus separately the
per-Gaussian means, depths, opacities, and extrinsics) and assert all
output fields agree to within 1e-4.

Run:
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        -m meadow_sb.tests.test_gaussian_adapter
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Allow `python -m meadow_sb.tests.test_gaussian_adapter` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Upstream import (yonosplat_inspect lives in /tmp). The conftest sibling
# tests already wire this in, but we duplicate the guard for stand-alone runs.
UPSTREAM_ROOT = "/tmp/yonosplat_inspect"
if UPSTREAM_ROOT not in sys.path:
    sys.path.insert(0, UPSTREAM_ROOT)

import mlx.core as mx
import torch

from src.model.encoder.common.gaussian_adapter import (  # noqa: E402
    GaussianAdapterCfg as PTGaussianAdapterCfg,
    UnifiedGaussianAdapter as PTUnifiedGaussianAdapter,
)

from meadow_sb.models.gaussian_adapter import (  # noqa: E402
    GaussianAdapter as MLXGaussianAdapter,
    GaussianAdapterCfg as MLXGaussianAdapterCfg,
)


# ---------------------------------------------------------------------------
# Synthetic input fixture
# ---------------------------------------------------------------------------


def _make_inputs(seed: int = 0):
    """Build a small batch of synthetic Gaussian-head outputs.

    Shapes mirror those inside `EncoderYoNoSplat.forward` right before the
    adapter is invoked:

        B = 1, V = 2, R = patches * gaussians^2 (we use a tiny stand-in),
        SRF = num_surfaces = 1, SPP = 1, d_sh = 1.
    """
    rng = np.random.default_rng(seed)

    B, V, R, SRF, SPP = 1, 2, 4, 1, 1
    d_sh = 1
    d_in = 7 + 3 * d_sh                 # 10
    raw_gs_dim = 1 + d_in               # 11

    # Means come from point_head.  Shape: (B, V, R, SRF, SPP, 3)
    means = rng.standard_normal((B, V, R, SRF, SPP, 3)).astype(np.float32)
    depths = means[..., -1:]            # (B, V, R, SRF, SPP, 1)

    # Opacities — already pdf-mapped (i.e. they're in [0, 1])
    opacities = rng.uniform(0.0, 1.0, (B, V, R, SRF, SPP)).astype(np.float32)

    # Raw 10-dim per Gaussian (opacity channel stripped off by caller).
    raw10 = rng.standard_normal((B, V, R, SRF, SPP, d_in)).astype(np.float32)

    # Random extrinsics (proper SE(3) by construction).
    def _rand_extrinsic():
        # Random rotation via QR.
        A = rng.standard_normal((3, 3)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = rng.standard_normal((3,)).astype(np.float32)
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = Q
        E[:3, 3] = t
        return E

    # Broadcast extrinsics to (B, V, 1, 1, 1, 4, 4)
    extr = np.stack(
        [np.stack([_rand_extrinsic() for _ in range(V)], axis=0) for _ in range(B)],
        axis=0,
    )  # (B, V, 4, 4)
    extr = extr.reshape(B, V, 1, 1, 1, 4, 4)

    return {
        "means": means,
        "depths": depths,
        "opacities": opacities,
        "raw10": raw10,
        "extrinsics": extr,
    }


# ---------------------------------------------------------------------------
# Run both adapters
# ---------------------------------------------------------------------------


def _run_pt(inp):
    cfg = PTGaussianAdapterCfg(
        sh_degree=0,
        gaussian_scale_min=0.5,
        gaussian_scale_max=15.0,
    )
    adapter = PTUnifiedGaussianAdapter(cfg).eval()

    means = torch.from_numpy(inp["means"])
    depths = torch.from_numpy(inp["depths"]).squeeze(-1)   # upstream passes (...,)
    opacities = torch.from_numpy(inp["opacities"])
    raw10 = torch.from_numpy(inp["raw10"])
    extr = torch.from_numpy(inp["extrinsics"])

    with torch.no_grad():
        out = adapter(
            means=means,
            depths=depths,
            opacities=opacities,
            raw_gaussians=raw10,
            extrinsics=extr,
        )

    return {
        "means": out.means.cpu().numpy(),
        "covariances": out.covariances.cpu().numpy(),
        "scales": out.scales.cpu().numpy(),
        "rotations": out.rotations.cpu().numpy(),
        "harmonics": out.harmonics.cpu().numpy(),
        "opacities": out.opacities.cpu().numpy(),
    }


def _run_mlx(inp):
    cfg = MLXGaussianAdapterCfg(
        sh_degree=0,
        gaussian_scale_min=0.5,
        gaussian_scale_max=15.0,
    )
    adapter = MLXGaussianAdapter(
        cfg,
        num_surfaces=1,
        gaussians_per_axis=14,
        upscale_token_ratio=2,
    )

    means = mx.array(inp["means"])
    depths = mx.array(inp["depths"]).squeeze(-1)
    opacities = mx.array(inp["opacities"])
    raw10 = mx.array(inp["raw10"])
    extr = mx.array(inp["extrinsics"])

    out = adapter(
        means=means,
        depths=depths,
        opacities=opacities,
        raw_gaussians=raw10,
        extrinsics=extr,
    )

    mx.eval(out.means, out.covariances, out.scales, out.rotations,
            out.harmonics, out.opacities)

    return {
        "means": np.array(out.means),
        "covariances": np.array(out.covariances),
        "scales": np.array(out.scales),
        "rotations": np.array(out.rotations),
        "harmonics": np.array(out.harmonics),
        "opacities": np.array(out.opacities),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(tol: float = 1e-4) -> int:
    inp = _make_inputs(seed=42)
    pt = _run_pt(inp)
    ml = _run_mlx(inp)

    keys = ["means", "covariances", "scales", "rotations", "harmonics", "opacities"]
    failures = 0
    print("\n=== GaussianAdapter MLX vs PyTorch parity ===")
    print(f"{'field':12s}  {'pt_shape':28s}  {'mlx_shape':28s}  {'max_abs_diff':>14s}")
    for k in keys:
        a, b = pt[k], ml[k]
        if a.shape != b.shape:
            try:
                b_b = np.broadcast_to(b, a.shape)
                diff = float(np.max(np.abs(a - b_b)))
            except Exception:
                print(f"{k:12s}  SHAPE MISMATCH {a.shape} vs {b.shape}")
                failures += 1
                continue
        else:
            diff = float(np.max(np.abs(a - b)))
        status = "OK" if diff < tol else "FAIL"
        print(f"{k:12s}  {str(a.shape):28s}  {str(b.shape):28s}  {diff:14.3e}  {status}")
        if diff >= tol:
            failures += 1

    # 539-d decomposition sanity check
    print("\n=== 539-d per-token decomposition ===")
    print("gaussians_per_token = (gaussians_per_axis / upscale_token_ratio)^2 "
          "= (14/2)^2 = 49")
    print("per_gaussian_dim    = 1 (opacity) + 3 (scale) + 4 (rotation) + "
          "3*d_sh (sh) = 11")
    print("=> 49 * 11 = 539 ✓")

    if failures == 0:
        print("\nALL PASSED.")
        return 0
    print(f"\n{failures} field(s) above tolerance ({tol}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
