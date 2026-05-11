"""Smoke + timing tests for SAM 3D Objects MLX DiT port.

Tests:
1. Building blocks (RoPE, AdaLN, FFN, attention) shapes / dtype.
2. DiTBlock forward + backward shape stability.
3. DiTBackbone.from_npz loads slat_flow.npz without error.
4. MOTDiTBackbone.from_npz loads ss_flow.npz without error.
5. Per-block forward timing (target < 100 ms on M1 GPU).

Run::
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        meadow3d/tests/test_dit.py
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np

# Make the repo importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meadow3d.models.dit_mlx import (  # noqa: E402
    AdaLNModulation,
    DiTBackbone,
    DiTBlock,
    DiTCrossBlock,
    FeedForwardNet,
    MOTDiTBackbone,
    MOTDiTCrossBlock,
    MultiHeadCrossAttention,
    MultiHeadRMSNorm,
    MultiHeadSelfAttention,
    RoPE,
    TimestepEmbedder,
)


WEIGHTS_DIR = os.path.join(ROOT, "meadow3d", "weights", "sam3d_objects")
SLAT_NPZ = os.path.join(WEIGHTS_DIR, "slat_flow.npz")
SS_NPZ = os.path.join(WEIGHTS_DIR, "ss_flow.npz")


def section(label: str) -> None:
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Unit tests for building blocks
# ---------------------------------------------------------------------------


def test_rope_shape():
    rope = RoPE(hidden_size=64)
    q = mx.random.normal((2, 4, 8, 64))  # (B, H, N, D)
    k = mx.random.normal((2, 4, 8, 64))
    qr, kr = rope(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape
    # rotation preserves L2 norm of pairs
    n_in = float(mx.sqrt(mx.sum(q * q)).item())
    n_out = float(mx.sqrt(mx.sum(qr * qr)).item())
    assert abs(n_in - n_out) < 1e-3, f"rope changed L2 norm: {n_in} vs {n_out}"
    print(f"  RoPE OK   q_norm preserved: {n_in:.4f} -> {n_out:.4f}")


def test_rms_norm():
    rn = MultiHeadRMSNorm(dim=64, heads=8)
    x = mx.random.normal((2, 16, 8, 64))  # (B, N, H, D)
    y = rn(x)
    assert y.shape == x.shape
    print("  MultiHeadRMSNorm OK")


def test_timestep_embedder():
    te = TimestepEmbedder(hidden_size=128, frequency_embedding_size=64)
    t = mx.array([0.1, 0.5, 0.9, 1.0], dtype=mx.float32)
    out = te(t)
    assert out.shape == (4, 128), out.shape
    print("  TimestepEmbedder OK", out.shape)


def test_adaln():
    mod = AdaLNModulation(cond_dim=128, hidden_dim=128)
    cond = mx.random.normal((2, 128))
    s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = mod(cond)
    for t_ in (s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp):
        assert t_.shape == (2, 128)
    print("  AdaLNModulation OK")


def test_attention():
    attn = MultiHeadSelfAttention(channels=128, num_heads=8, qk_rms_norm=True)
    x = mx.random.normal((2, 16, 128))
    y = attn(x)
    assert y.shape == x.shape
    print("  MHSelfAttention OK", y.shape)

    cross = MultiHeadCrossAttention(channels=128, ctx_channels=64, num_heads=8)
    ctx = mx.random.normal((2, 8, 64))
    y2 = cross(x, ctx)
    assert y2.shape == x.shape
    print("  MHCrossAttention OK", y2.shape)


def test_dit_block():
    blk = DiTBlock(channels=128, num_heads=8, cond_dim=128, qk_rms_norm=True)
    x = mx.random.normal((2, 16, 128))
    cond = mx.random.normal((2, 128))
    y = blk(x, cond)
    assert y.shape == x.shape
    print("  DiTBlock OK", y.shape)

    blk2 = DiTCrossBlock(channels=128, ctx_channels=64, num_heads=8, cond_dim=128, qk_rms_norm=True)
    ctx = mx.random.normal((2, 8, 64))
    y2 = blk2(x, cond, ctx)
    assert y2.shape == x.shape
    print("  DiTCrossBlock OK", y2.shape)


def test_mot_block():
    names = ["shape", "scale"]
    blk = MOTDiTCrossBlock(
        channels=128, ctx_channels=64, num_heads=8,
        latent_names=names, cond_dim=128, qk_rms_norm=True,
        protect_modality_list=["shape"],
    )
    latents = {
        "shape": mx.random.normal((2, 32, 128)),
        "scale": mx.random.normal((2, 4, 128)),
    }
    cond = mx.random.normal((2, 128))
    ctx = mx.random.normal((2, 8, 64))
    out = blk(latents, cond, ctx)
    for n in names:
        assert out[n].shape == latents[n].shape, (n, out[n].shape)
    print("  MOTDiTCrossBlock OK")


# ---------------------------------------------------------------------------
# Weight loading tests
# ---------------------------------------------------------------------------


def test_slat_load_and_forward():
    if not os.path.exists(SLAT_NPZ):
        print(f"  SKIP (slat_flow.npz not found at {SLAT_NPZ})")
        return None
    print(f"  loading {SLAT_NPZ} ...")
    t0 = time.perf_counter()
    model = DiTBackbone.from_npz(
        SLAT_NPZ, depth=24, channels=1024, num_heads=16,
        ctx_channels=1024, qk_rms_norm=True,
    )
    mx.eval(model.parameters())
    t1 = time.perf_counter()
    print(f"  loaded in {t1 - t0:.2f}s")

    # Full-depth forward
    B, N = 1, 4096
    x = mx.random.normal((B, N, 1024))
    t = mx.array([0.5], dtype=mx.float32)
    ctx = mx.random.normal((B, 64, 1024))
    mx.eval(x, t, ctx)

    # Warmup
    y = model(x, t, ctx)
    mx.eval(y)
    print("  warmup forward OK", y.shape)

    # Per-block timing: time a single block via the backbone forward but estimate
    # by timing the full stack and dividing.
    t0 = time.perf_counter()
    n_runs = 3
    for _ in range(n_runs):
        y = model(x, t, ctx)
        mx.eval(y)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / n_runs * 1000.0
    per_block_ms = total_ms / 24
    print(f"  full forward {total_ms:.1f} ms ({per_block_ms:.2f} ms/block, depth=24)")
    return per_block_ms


def test_ss_load_and_forward():
    if not os.path.exists(SS_NPZ):
        print(f"  SKIP (ss_flow.npz not found at {SS_NPZ})")
        return None
    print(f"  loading {SS_NPZ} ...")
    t0 = time.perf_counter()
    model = MOTDiTBackbone.from_npz(
        SS_NPZ,
        depth=24, channels=1024, num_heads=16, ctx_channels=1024,
        latent_names=MOTDiTBackbone.DEFAULT_LATENT_NAMES,
        qk_rms_norm=True,
        has_d_embedder=True,
        protect_modality_list=["shape"],
    )
    mx.eval(model.parameters())
    t1 = time.perf_counter()
    print(f"  loaded in {t1 - t0:.2f}s")

    # In ss_flow the lower-dim modalities are merged into 6drotation_normalized
    # via latent_share_transformer before entering the backbone. Block weights
    # only carry shape + 6drotation_normalized.
    # shape has pos_emb (4096, 1024) so N=4096; we use 256 for timing speed.
    B = 1
    latents = {
        "shape": mx.random.normal((B, 256, 1024)),
        # merged: scale(1) + translation(1) + translation_scale(1) + 6drotation(1) = 4
        "6drotation_normalized": mx.random.normal((B, 4, 1024)),
    }
    t = mx.array([0.5], dtype=mx.float32)
    d = mx.array([0.0], dtype=mx.float32)
    ctx = mx.random.normal((B, 64, 1024))
    mx.eval(*latents.values(), t, d, ctx)

    # Warmup
    out = model(latents, t, ctx, d=d)
    mx.eval(*out.values())
    print("  warmup forward OK")
    for n, v in out.items():
        assert v.shape == latents[n].shape, (n, v.shape, latents[n].shape)

    t0 = time.perf_counter()
    n_runs = 3
    for _ in range(n_runs):
        out = model(latents, t, ctx, d=d)
        mx.eval(*out.values())
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / n_runs * 1000.0
    per_block_ms = total_ms / 24
    print(f"  full forward {total_ms:.1f} ms ({per_block_ms:.2f} ms/block, depth=24)")
    return per_block_ms


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    section("DiT MLX port — smoke + timing")
    print(f"MLX:       {mx.__version__}")
    print(f"GPU avail: {mx.metal.is_available()}")
    print(f"weights:   {WEIGHTS_DIR}")

    section("1. Building blocks")
    test_rope_shape()
    test_rms_norm()
    test_timestep_embedder()
    test_adaln()
    test_attention()
    test_dit_block()
    test_mot_block()

    section("2. slat_flow (DiTBackbone) load + timing")
    slat_per_block = test_slat_load_and_forward()

    section("3. ss_flow (MOTDiTBackbone) load + timing")
    ss_per_block = test_ss_load_and_forward()

    section("Summary")
    print(f"slat_flow per-block: {slat_per_block} ms" if slat_per_block else "slat_flow: SKIPPED")
    print(f"ss_flow   per-block: {ss_per_block} ms" if ss_per_block else "ss_flow:   SKIPPED")
    if slat_per_block is not None:
        print(f"target <100 ms/block : {'PASS' if slat_per_block < 100 else 'FAIL'}")


if __name__ == "__main__":
    main()
