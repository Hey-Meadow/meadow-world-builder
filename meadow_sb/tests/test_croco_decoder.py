"""Per-block numerical parity tests for the CroCo decoder MLX port.

Loads each PT-side activation dump (`dec_block_00.npz` ... `dec_block_35.npz`)
and asserts that the MLX `BlockRope` produces the same output, given the same
input and the reconstructed `xpos` grid.

Gate: `max(|mlx_out - pt_out|) < 1e-3` per block. Documented per-block diffs
are printed at the bottom of the run.

Test input:
    - dec_block_NN.npz stores 'in' (B', N', C) and 'out' (B', N', C), where
      (B', N') alternates between (B*N=2, num_register+h*w=261) at even blocks
      and (B=1, N*(num_register+h*w)=522) at odd blocks for the bootstrap
      2-view 224x224 test_input.
    - We reconstruct `xpos` deterministically from `PositionGetter` rules.

Weights:
    - `re10k_224x224_ctx2to32.ckpt` is a Lightning checkpoint; we use torch to
      load it on CPU (no MPS / CUDA needed) and slice by prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

# Repo paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DUMPS = REPO_ROOT / "research" / "yonosplat_bootstrap" / "dumps" / "per_block"
WEIGHTS = REPO_ROOT / "research" / "yonosplat_bootstrap" / "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"

sys.path.insert(0, str(REPO_ROOT))
from meadow_sb.models.croco_decoder import (  # noqa: E402
    BlockRope,
    RoPE2D,
    build_decoder_positions,
    load_block_weights_from_pt,
)


# Hyper-parameters (matched against state_dict + dump shapes).
DIM = 1024
NUM_HEADS = 16
MLP_RATIO = 4.0
ROPE_FREQ = 100.0
NUM_REG = 5
H = W = 224
PATCH = 14
B, N = 1, 2      # test_input.npz is 2-view, batch=1.
hw_per_view = NUM_REG + (H // PATCH) * (W // PATCH)   # = 5 + 256 = 261


# ------------------------------------------------------------------------------
# Lazy fixtures: load PT state-dict + the full pos grid once for all tests.
# ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pt_sd():
    """Slice of the YoNoSplat state-dict containing only decoder weights and
    `encoder.backbone.register_token`. Returns a `dict[str, np.ndarray]`.
    """
    import torch

    if not WEIGHTS.exists():
        pytest.skip(f"checkpoint not found: {WEIGHTS}")
    ckpt = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
    full = ckpt["state_dict"]
    sd: dict[str, np.ndarray] = {}
    for k, v in full.items():
        if k.startswith("encoder.backbone.decoder.") or k == "encoder.backbone.register_token":
            sd[k] = v.detach().cpu().float().numpy()
    return sd


@pytest.fixture(scope="module")
def base_pos():
    """The `(B*N, hw_per_view, 2)` position grid before any reshape."""
    return build_decoder_positions(B, N, H // PATCH, W // PATCH, NUM_REG)


@pytest.fixture(scope="module")
def rope():
    return RoPE2D(freq=ROPE_FREQ)


# ------------------------------------------------------------------------------
# Per-block test
# ------------------------------------------------------------------------------


PER_BLOCK_DIFFS: dict[int, float] = {}


@pytest.mark.parametrize("block_idx", list(range(36)))
def test_block_matches_pt(block_idx, pt_sd, base_pos, rope):
    """Each block: load PT weights, reshape pos for even/odd, run MLX, compare."""

    dump_path = DUMPS / f"dec_block_{block_idx:02d}.npz"
    if not dump_path.exists():
        pytest.skip(f"dump missing: {dump_path}")
    d = np.load(dump_path)
    x_np = d["in"]
    y_ref = d["out"]

    # Build the MLX block + load weights.
    blk = BlockRope(dim=DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, rope=rope)
    load_block_weights_from_pt(blk, pt_sd, f"encoder.backbone.decoder.{block_idx}.")

    # Reshape xpos to match the input batching of this block.
    # Even block -> (B*N, hw_per_view, 2). Odd block -> (B, N*hw_per_view, 2).
    if block_idx % 2 == 0:
        xpos = base_pos.reshape(B * N, hw_per_view, 2)
        assert tuple(x_np.shape) == (B * N, hw_per_view, DIM), (
            f"block {block_idx} input shape {x_np.shape} != expected {(B*N, hw_per_view, DIM)}"
        )
    else:
        xpos = base_pos.reshape(B, N * hw_per_view, 2)
        assert tuple(x_np.shape) == (B, N * hw_per_view, DIM), (
            f"block {block_idx} input shape {x_np.shape} != expected {(B, N*hw_per_view, DIM)}"
        )

    x = mx.array(x_np)
    y = blk(x, xpos=xpos)
    mx.eval(y)
    y_np = np.array(y)

    diff = float(np.max(np.abs(y_np - y_ref)))
    PER_BLOCK_DIFFS[block_idx] = diff
    # Print for the diff table (pytest captures stdout — use -s to see).
    print(f"[dec_block_{block_idx:02d}] max|mlx - pt| = {diff:.3e}")
    assert diff < 1e-3, f"block {block_idx} max-diff {diff:.3e} exceeds 1e-3"


def test_summary():
    """Print the per-block diff table at the end of the run."""
    if not PER_BLOCK_DIFFS:
        pytest.skip("no per-block diffs collected")
    print("\nPer-block max-diff table (mlx vs pt):")
    print("  block | max|diff|")
    print("  ------+-----------")
    passed = 0
    for i in sorted(PER_BLOCK_DIFFS):
        d = PER_BLOCK_DIFFS[i]
        ok = "OK" if d < 1e-3 else "FAIL"
        print(f"   {i:>3d}  | {d:.3e}  {ok}")
        if d < 1e-3:
            passed += 1
    print(f"\nPassed: {passed}/{len(PER_BLOCK_DIFFS)}")
