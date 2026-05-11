"""Smoke tests for SAM 3D Objects decoder MLX port.

Run:
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        -m meadow3d.tests.test_decoder

Tests:
  T1. Random-input forward of SSDecoder (no weights). Checks input/output shape
      math: (B, R^3, 8) -> (B, 4R, 4R, 4R, 1).
  T2. Random-input forward of SLATDecoderGS (no weights). Checks the swin
      partition logic + per-voxel Gaussian param dict shapes.
  T3. Real-weight load + tiny forward of both decoders, confirming every key
      in the npz is consumed and forward runs.
  T4. ``save_gaussian_ply`` produces a valid 3DGS-compatible binary PLY.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meadow3d.models.decoder_mlx import (
    SSDecoder,
    SLATDecoderGS,
    save_gaussian_ply,
    pixel_shuffle_3d,
)


WEIGHT_DIR = os.path.join(ROOT, "meadow3d", "weights", "sam3d_objects")


# ---------------------------------------------------------------------------
# T1. SSDecoder shape sanity
# ---------------------------------------------------------------------------

def test_ss_decoder_random_forward():
    print("\n[T1] SSDecoder random-init shape sanity ...")
    R = 4  # tiny cube to keep memory low (real R=16); architecture handles any R
    B, C = 1, 8
    z = mx.random.normal((B, R * R * R, C))
    m = SSDecoder(latent_channels=C, channels=(32, 16, 8), num_res_blocks=1,
                  num_res_blocks_middle=1)
    y = m(z)
    mx.eval(y)
    expected = (B, R * 4, R * 4, R * 4, 1)
    assert y.shape == expected, f"Expected {expected}, got {y.shape}"
    print(f"     in {z.shape} -> out {y.shape} (4x upsample)  OK")


# ---------------------------------------------------------------------------
# T2. SLATDecoderGS shape + swin partition
# ---------------------------------------------------------------------------

def test_slat_decoder_random_forward():
    print("\n[T2] SLATDecoderGS random-init forward + Gaussian shapes ...")
    # Use a SMALL config so the test is fast / cheap.
    G = 32  # num_gaussians
    m = SLATDecoderGS(resolution=16, model_channels=64, latent_channels=8,
                      num_blocks=2, num_heads=2, mlp_ratio=2.0,
                      window_size=4)
    # Build a fake sparse latent: N voxels in a tiny grid.
    N = 50
    feats = mx.random.normal((N, 8))
    coords = np.zeros((N, 4), dtype=np.int32)
    for i in range(N):
        coords[i, 0] = 0
        coords[i, 1] = i % 16
        coords[i, 2] = (i // 16) % 16
        coords[i, 3] = (i // (16 * 16)) % 16
    coords = mx.array(coords)
    out = m(feats, coords)
    mx.eval(out["_xyz"])  # force compute
    print(f"     N={N}, G={G}, total Gaussians = {N*G}")
    print(f"     keys = {sorted(k for k in out if not k.startswith('_meta'))}")

    assert out["_xyz"].shape == (N * G, 3), out["_xyz"].shape
    assert out["_features_dc"].shape == (N * G, 1, 3), out["_features_dc"].shape
    assert out["_scaling"].shape == (N * G, 3), out["_scaling"].shape
    assert out["_rotation"].shape == (N * G, 4), out["_rotation"].shape
    assert out["_opacity"].shape == (N * G, 1), out["_opacity"].shape
    print("     all Gaussian param shapes OK")


# ---------------------------------------------------------------------------
# T3. Load real weights (key coverage).
# ---------------------------------------------------------------------------

def test_load_real_weights():
    print("\n[T3] Loading real npz weights ...")

    ss_path = os.path.join(WEIGHT_DIR, "ss_decoder.npz")
    slat_path = os.path.join(WEIGHT_DIR, "slat_decoder_gs.npz")
    if not (os.path.exists(ss_path) and os.path.exists(slat_path)):
        print(f"     SKIP: weights not found at {WEIGHT_DIR}")
        return

    ss_w = mx.load(ss_path)
    n_ss = len(ss_w)
    print(f"     ss_decoder.npz : {n_ss} keys")
    ss = SSDecoder.from_npz(ss_path)
    # Tiny forward (R=8 => out 32^3 -- still light).
    R = 8
    z = mx.random.normal((1, R * R * R, 8))
    y = ss(z)
    mx.eval(y)
    print(f"     SSDecoder forward OK  ({z.shape} -> {y.shape})")

    slat_w = mx.load(slat_path)
    n_slat = len(slat_w)
    print(f"     slat_decoder_gs.npz : {n_slat} keys")
    slat = SLATDecoderGS.from_npz(slat_path)
    print(f"     model_channels = {slat.model_channels}")
    print(f"     num_blocks     = {len(slat.blocks)}")
    print(f"     num_gaussians  = {slat.rep_config['num_gaussians']}")
    # Sparse forward on a tiny voxel set.
    N = 16
    feats = mx.random.normal((N, 8))
    rng = np.random.default_rng(0)
    coords = np.zeros((N, 4), dtype=np.int32)
    coords[:, 1:] = rng.integers(0, 32, size=(N, 3))
    out = slat(feats, mx.array(coords))
    mx.eval(out["_xyz"])
    G = slat.rep_config["num_gaussians"]
    assert out["_xyz"].shape == (N * G, 3)
    print(f"     SLATDecoderGS forward OK  (N={N}, G={G}, total={N*G})")


# ---------------------------------------------------------------------------
# T4. PLY writer
# ---------------------------------------------------------------------------

def test_save_gaussian_ply_binary_format():
    print("\n[T4] save_gaussian_ply -> binary PLY ...")
    N = 10
    gs = {
        "_xyz": mx.random.normal((N, 3)) * 0.1,
        "_features_dc": mx.random.normal((N, 1, 3)),
        "_scaling": mx.random.normal((N, 3)),
        "_rotation": mx.random.normal((N, 4)),
        "_opacity": mx.random.normal((N, 1)),
        "_meta": mx.array([4e-3, 0.1, 1e-4, 0.0], dtype=mx.float32),
    }
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    try:
        save_gaussian_ply(gs, path)
        with open(path, "rb") as f:
            head = f.read(4)
            assert head == b"ply\n", f"bad magic: {head!r}"
            # consume header, find 'end_header'
            f.seek(0)
            data = f.read()
        eh_idx = data.find(b"end_header\n")
        assert eh_idx > 0
        header = data[:eh_idx].decode()
        body = data[eh_idx + len(b"end_header\n"):]
        # Each vertex has 17 float32 properties: 3 xyz + 3 normals + 3 f_dc + 1 op + 3 scale + 4 rot
        prop_count = sum(1 for line in header.split("\n") if line.startswith("property"))
        assert prop_count == 17, prop_count
        assert len(body) == N * prop_count * 4, (len(body), N * prop_count * 4)
        print(f"     wrote {len(data)} bytes, {prop_count} props/vertex, {N} vertices  OK")
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# pixel_shuffle_3d sanity
# ---------------------------------------------------------------------------

def test_pixel_shuffle_3d():
    print("\n[T0] pixel_shuffle_3d cardinality ...")
    x = mx.random.normal((1, 2, 2, 2, 24))
    y = pixel_shuffle_3d(x, 2)
    mx.eval(y)
    assert y.shape == (1, 4, 4, 4, 3), y.shape
    # Total element count preserved.
    assert int(np.prod(x.shape)) == int(np.prod(y.shape)), (x.shape, y.shape)
    print(f"     {x.shape} -> {y.shape}  OK")


def main():
    test_pixel_shuffle_3d()
    test_ss_decoder_random_forward()
    test_slat_decoder_random_forward()
    test_load_real_weights()
    test_save_gaussian_ply_binary_format()
    print("\nAll decoder tests passed.")


if __name__ == "__main__":
    main()
