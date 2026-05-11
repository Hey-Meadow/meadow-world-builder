"""Smoke tests for `meadow3d.models.embedders_mlx`.

Validates:
1. ConditionEmbedder.from_npz() loads ss_embedder.npz and slat_embedder.npz.
2. Forward produces the expected token shape (B, N_tokens, embed_dim).
3. TimeEmbedding produces shape (B, hidden_size).

Run:
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        -m pytest meadow3d/tests/test_embedders.py -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from meadow3d.models.embedders_mlx import (  # noqa: E402
    ConditionEmbedder,
    ImageEmbedder,
    PointPatchEmbedder,
    TimeEmbedding,
)


SS_NPZ = REPO_ROOT / "meadow3d" / "weights" / "sam3d_objects" / "ss_embedder.npz"
SLAT_NPZ = REPO_ROOT / "meadow3d" / "weights" / "sam3d_objects" / "slat_embedder.npz"


# ---------------------------------------------------------------------------
# Standalone module shape tests (do not require weights).
# ---------------------------------------------------------------------------


def test_time_embedding_shape():
    te = TimeEmbedding(hidden_size=1024, frequency_embedding_size=256)
    t = mx.array([0.0, 0.5, 1.0])
    out = te(t)
    assert out.shape == (3, 1024), f"got {out.shape}"


def test_image_embedder_shape():
    """Forward pass without trained weights -- just checks shape contract."""
    emb = ImageEmbedder(
        input_size=518, patch_size=14, embed_dim=1024,
        depth=2,  # tiny for speed
        num_heads=16, num_register_tokens=4,
    )
    img = mx.zeros((1, 518, 518, 3))
    out = emb(img)
    # 1 cls + 37*37 patches = 1 + 1369 = 1370
    assert out.shape == (1, 1370, 1024), f"got {out.shape}"


def test_point_patch_embedder_shape():
    pe = PointPatchEmbedder(
        input_size=256, patch_size=8, embed_dim=512, num_heads=16, depth=1
    )
    # PT input convention: (B, 3, H, W). Use H=W=256 to match input_size and
    # provide finite values so the valid_mask is all-True.
    pts = mx.ones((1, 3, 256, 256)) * 0.1
    out = pe(pts)
    # 256/8 = 32 windows per side -> 1024 tokens, embed_dim=512
    assert out.shape == (1, 32 * 32, 512), f"got {out.shape}"


# ---------------------------------------------------------------------------
# Weight-loading tests (skipped if npz absent).
# ---------------------------------------------------------------------------


def _ss_inputs():
    img = mx.zeros((1, 518, 518, 3))
    rgb_image = mx.zeros((1, 518, 518, 3))
    mask = mx.zeros((1, 518, 518, 1))
    rgb_image_mask = mx.zeros((1, 518, 518, 1))
    pointmap = mx.ones((1, 3, 256, 256)) * 0.1  # finite -> valid_mask all True
    rgb_pointmap = mx.zeros((1, 3, 256, 256))
    return dict(
        image=img,
        rgb_image=rgb_image,
        mask=mask,
        rgb_image_mask=rgb_image_mask,
        pointmap=pointmap,
        rgb_pointmap=rgb_pointmap,
    )


@pytest.mark.skipif(not SS_NPZ.exists(), reason="ss_embedder.npz missing; run convert.py")
def test_load_ss_condition_embedder():
    cond = ConditionEmbedder.from_npz(str(SS_NPZ))
    assert cond.embed_dim == 1024
    # Six kwargs (2 per embedder) -> 4 image trunks + 2 point trunks
    out = cond(**_ss_inputs())
    expected_n = 4 * (1 + 37 * 37) + 2 * (32 * 32)
    assert out.shape == (1, expected_n, 1024), (
        f"expected (1, {expected_n}, 1024) got {out.shape}"
    )
    print(f"[ok] ss_condition_embedder out shape = {out.shape}")


@pytest.mark.skipif(not SLAT_NPZ.exists(), reason="slat_embedder.npz missing")
def test_load_slat_condition_embedder():
    cond = ConditionEmbedder.from_npz(str(SLAT_NPZ))
    assert cond.embed_dim == 1024
    inputs = _ss_inputs()
    inputs.pop("pointmap")
    inputs.pop("rgb_pointmap")
    out = cond(**inputs)
    expected_n = 4 * (1 + 37 * 37)
    assert out.shape == (1, expected_n, 1024), (
        f"expected (1, {expected_n}, 1024) got {out.shape}"
    )
    print(f"[ok] slat_condition_embedder out shape = {out.shape}")


# ---------------------------------------------------------------------------
# Differentiability sanity (different inputs -> different outputs).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SS_NPZ.exists(), reason="ss_embedder.npz missing")
def test_input_sensitivity():
    """Different image conditioning produces different output tokens."""
    cond = ConditionEmbedder.from_npz(str(SS_NPZ))
    rng = np.random.default_rng(0)

    inputs_a = _ss_inputs()
    inputs_b = _ss_inputs()
    inputs_a["image"] = mx.array(rng.standard_normal((1, 518, 518, 3)).astype(np.float32))
    inputs_b["image"] = mx.array(rng.standard_normal((1, 518, 518, 3)).astype(np.float32))

    out_a = cond(**inputs_a)
    out_b = cond(**inputs_b)

    diff = float(mx.mean(mx.abs(out_a - out_b)))
    assert diff > 1e-3, f"different inputs should produce different outputs; got mean abs diff = {diff}"
    print(f"[ok] input-sensitivity mean abs diff = {diff:.4f}")


if __name__ == "__main__":
    test_time_embedding_shape()
    test_image_embedder_shape()
    test_point_patch_embedder_shape()
    print("[ok] standalone shape tests passed")
    if SS_NPZ.exists():
        test_load_ss_condition_embedder()
        test_input_sensitivity()
    if SLAT_NPZ.exists():
        test_load_slat_condition_embedder()
    print("ALL TESTS PASSED")
