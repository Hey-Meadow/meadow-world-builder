"""Correctness + timing test for ``SparseSubmConv3d``.

Run with:

    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        /Users/akaihuangm1/Desktop/github/sam-3d-objects/meadow3d/tests/test_sparse_subm_conv3d.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Make the repo importable when run directly.
_REPO = "/Users/akaihuangm1/Desktop/github/sam-3d-objects"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mlx.core as mx  # noqa: E402

from meadow3d.kernels.sparse_subm_conv3d import (  # noqa: E402
    SparseSubmConv3d,
    build_neighbor_table,
    clear_neighbor_cache,
    submconv3d_metal,
    submconv3d_mlx,
)


# ---------------------------------------------------------------------------
# Numpy reference (correct, slow, no spconv dependency).
# ---------------------------------------------------------------------------


def numpy_submconv3d_reference(
    features: np.ndarray,      # (N, C_in)
    coords: np.ndarray,        # (N, 4) int32 (b, z, y, x)
    weight: np.ndarray,        # (K, K, K, C_in, C_out)
    bias: np.ndarray | None,   # (C_out,) or None
) -> np.ndarray:
    """Brutally explicit numpy implementation matching spconv SubMConv3d."""
    N, C_in = features.shape
    K, _, _, _, C_out = weight.shape
    half = K // 2

    coord2row: dict[tuple[int, int, int, int], int] = {}
    for n in range(N):
        b, z, y, x = coords[n]
        coord2row[(int(b), int(z), int(y), int(x))] = n

    out = np.zeros((N, C_out), dtype=np.float32)
    for n in range(N):
        b, z, y, x = coords[n]
        for dz in range(-half, half + 1):
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    nb = coord2row.get((int(b), int(z + dz), int(y + dy), int(x + dx)))
                    if nb is None:
                        continue
                    # weight[dz+half, dy+half, dx+half, :, :]
                    w = weight[dz + half, dy + half, dx + half]  # (C_in, C_out)
                    out[n] += features[nb] @ w
    if bias is not None:
        out += bias
    return out


# ---------------------------------------------------------------------------
# Test data.
# ---------------------------------------------------------------------------


def make_random_voxels(N: int, grid: int, seed: int = 0) -> np.ndarray:
    """N unique voxel coords in batch=0 within a `grid`^3 box."""
    rng = np.random.default_rng(seed)
    used: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    while len(out) < N:
        z = int(rng.integers(0, grid))
        y = int(rng.integers(0, grid))
        x = int(rng.integers(0, grid))
        key = (z, y, x)
        if key in used:
            continue
        used.add(key)
        out.append((0, z, y, x))
    return np.asarray(out, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_correctness(N: int = 1024, C_in: int = 32, C_out: int = 64, seed: int = 0,
                     verbose: bool = True) -> dict:
    """Compare MLX gather+einsum and Metal kernel vs numpy reference."""
    rng = np.random.default_rng(seed)
    coords_np = make_random_voxels(N, grid=24, seed=seed)
    feats_np = rng.standard_normal((N, C_in), dtype=np.float32)
    weight_np = (rng.standard_normal((3, 3, 3, C_in, C_out), dtype=np.float32) * 0.1).astype(np.float32)
    bias_np = (rng.standard_normal((C_out,), dtype=np.float32) * 0.05).astype(np.float32)

    # --- numpy reference
    t0 = time.perf_counter()
    ref = numpy_submconv3d_reference(feats_np, coords_np, weight_np, bias_np)
    t_ref = time.perf_counter() - t0

    # --- MLX path
    feats_mx = mx.array(feats_np)
    coords_mx = mx.array(coords_np)
    weight_mx = mx.array(weight_np.reshape(27, C_in, C_out))
    bias_mx = mx.array(bias_np)

    layer = SparseSubmConv3d(C_in, C_out, kernel_size=3, bias=True, backend="mlx")
    layer.load_weight(weight_mx, bias_mx)

    out_mlx = layer(feats_mx, coords_mx)
    mx.eval(out_mlx)
    out_mlx_np = np.asarray(out_mlx)

    diff_mlx = float(np.max(np.abs(out_mlx_np - ref)))

    # --- Metal kernel
    layer_metal = SparseSubmConv3d(C_in, C_out, kernel_size=3, bias=True, backend="metal")
    layer_metal.load_weight(weight_mx, bias_mx)
    out_metal = layer_metal(feats_mx, coords_mx)
    mx.eval(out_metal)
    out_metal_np = np.asarray(out_metal)
    diff_metal = float(np.max(np.abs(out_metal_np - ref)))
    metal_ok = layer_metal.backend == "metal"

    if verbose:
        print(f"[correctness] N={N}, C_in={C_in}, C_out={C_out}")
        print(f"  numpy ref time:     {t_ref*1000:.1f} ms")
        print(f"  max_abs_diff (MLX):   {diff_mlx:.3e}")
        print(f"  max_abs_diff (Metal): {diff_metal:.3e} (compiled={metal_ok})")

    tol = 1e-3
    assert diff_mlx < tol, f"MLX backend diff {diff_mlx} exceeds tol {tol}"
    if metal_ok:
        assert diff_metal < tol, f"Metal backend diff {diff_metal} exceeds tol {tol}"
    return {
        "diff_mlx": diff_mlx,
        "diff_metal": diff_metal,
        "metal_compiled": metal_ok,
    }


def test_neighbor_cache():
    """Same coords array → cached neighbor table reused."""
    coords_np = make_random_voxels(256, grid=16, seed=1)
    coords_mx = mx.array(coords_np)
    clear_neighbor_cache()

    t0 = time.perf_counter()
    nt1 = build_neighbor_table(coords_mx, 3)
    mx.eval(nt1)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    nt2 = build_neighbor_table(coords_mx, 3)
    mx.eval(nt2)
    t_second = time.perf_counter() - t0

    assert nt1 is nt2 or np.array_equal(np.asarray(nt1), np.asarray(nt2))
    print(f"[cache] first build: {t_first*1000:.2f} ms ; cached lookup: {t_second*1e6:.1f} µs")
    assert t_second < t_first / 5, "cache must be much faster than rebuild"


def bench(N: int = 4096, C_in: int = 128, C_out: int = 128, repeats: int = 50):
    """Per-call timing for the slat hot-path regime.

    Real model dimensions (from ``structured_latent_flow.py``):
        - patch_size=2, slat grid up to 32^3 → up to ~16k voxels (typically 2-8k)
        - io block channels: 128, 256, 512 (the 128 case dominates calls).
    """
    rng = np.random.default_rng(7)
    coords_np = make_random_voxels(N, grid=int(np.cbrt(N) * 1.5) + 4, seed=7)
    feats_np = rng.standard_normal((N, C_in), dtype=np.float32)
    weight_np = (rng.standard_normal((27, C_in, C_out), dtype=np.float32) * 0.05).astype(np.float32)
    bias_np = np.zeros((C_out,), dtype=np.float32)

    feats_mx = mx.array(feats_np)
    coords_mx = mx.array(coords_np)
    weight_mx = mx.array(weight_np)
    bias_mx = mx.array(bias_np)

    # Pre-build neighbor table once.
    clear_neighbor_cache()
    t0 = time.perf_counter()
    nt = build_neighbor_table(coords_mx, 3)
    mx.eval(nt)
    t_nt = time.perf_counter() - t0

    # MLX path warmup.
    out = submconv3d_mlx(feats_mx, nt, weight_mx, bias_mx); mx.eval(out)
    out = submconv3d_mlx(feats_mx, nt, weight_mx, bias_mx); mx.eval(out)

    # Per-call sync: forces actual GPU dispatch each iter so timings are honest.
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = submconv3d_mlx(feats_mx, nt, weight_mx, bias_mx)
        mx.eval(out)
    t_mlx = (time.perf_counter() - t0) / repeats

    # Metal path.
    metal_ok = True
    t_metal = float("nan")
    try:
        out_m = submconv3d_metal(feats_mx, nt, weight_mx, bias_mx); mx.eval(out_m)
        out_m = submconv3d_metal(feats_mx, nt, weight_mx, bias_mx); mx.eval(out_m)
        t0 = time.perf_counter()
        for _ in range(repeats):
            out_m = submconv3d_metal(feats_mx, nt, weight_mx, bias_mx)
            mx.eval(out_m)
        t_metal = (time.perf_counter() - t0) / repeats
    except Exception as e:
        metal_ok = False
        print(f"  metal kernel failed: {e!r}")

    print(f"[bench] N={N}, C_in={C_in}, C_out={C_out}, repeats={repeats}")
    print(f"  neighbor table build (one-shot): {t_nt*1000:.2f} ms")
    print(f"  MLX gather+einsum per-call:    {t_mlx*1000:.2f} ms")
    if metal_ok:
        speedup = t_mlx / t_metal if t_metal > 0 else float("nan")
        print(f"  Metal kernel per-call:         {t_metal*1000:.2f} ms  (×{speedup:.2f} vs MLX)")
    # 800 calls × cheap path projects total
    if metal_ok:
        projected = min(t_mlx, t_metal) * 800
    else:
        projected = t_mlx * 800
    print(f"  projected per-inference cost (800 calls): {projected:.2f} s")
    return {
        "t_nt": t_nt,
        "t_mlx": t_mlx,
        "t_metal": t_metal,
        "metal_ok": metal_ok,
    }


def test_cache_amortization():
    """Show that the dominant cost is amortized over 800 calls."""
    coords_np = make_random_voxels(2048, grid=20, seed=2)
    feats_np = np.random.default_rng(0).standard_normal((2048, 64), dtype=np.float32)
    weight_np = (np.random.default_rng(1).standard_normal((27, 64, 64), dtype=np.float32) * 0.05).astype(np.float32)

    feats_mx = mx.array(feats_np); coords_mx = mx.array(coords_np)
    weight_mx = mx.array(weight_np); bias_mx = mx.zeros((64,), dtype=mx.float32)
    layer = SparseSubmConv3d(64, 64, kernel_size=3, bias=True, backend="mlx")
    layer.load_weight(weight_mx, bias_mx)

    clear_neighbor_cache()

    # First call pays for neighbor build; subsequent calls don't.
    t0 = time.perf_counter()
    o = layer(feats_mx, coords_mx); mx.eval(o)
    t_first = time.perf_counter() - t0

    # warmup
    for _ in range(3):
        o = layer(feats_mx, coords_mx); mx.eval(o)

    t0 = time.perf_counter()
    for _ in range(20):
        o = layer(feats_mx, coords_mx)
        mx.eval(o)
    t_subseq = (time.perf_counter() - t0) / 20

    print(f"[amortization] first call (incl. neighbor build): {t_first*1000:.2f} ms ; "
          f"subsequent: {t_subseq*1000:.2f} ms")
    assert t_subseq < t_first, "cached calls must be faster than the first"


def test_zero_neighbors_safe():
    """If a voxel has no neighbors at all, output should be just bias."""
    # One isolated voxel far from any other.
    coords = np.array([[0, 50, 50, 50]], dtype=np.int32)
    feats = np.ones((1, 4), dtype=np.float32)
    weight = np.ones((27, 4, 8), dtype=np.float32) * 0.1
    bias = np.arange(8, dtype=np.float32)

    layer = SparseSubmConv3d(4, 8, kernel_size=3, bias=True, backend="mlx")
    layer.load_weight(mx.array(weight), mx.array(bias))
    out = layer(mx.array(feats), mx.array(coords))
    mx.eval(out)
    out_np = np.asarray(out)
    # Center kernel offset (k=13) IS the voxel itself; it's a neighbor.
    # So output = feats @ weight[13] + bias.
    expected = feats[0] @ weight[13] + bias
    diff = float(np.max(np.abs(out_np[0] - expected)))
    print(f"[isolated voxel] max_abs_diff: {diff:.3e}")
    assert diff < 1e-5


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("SparseSubmConv3d — correctness + timing")
    print("=" * 60)

    test_zero_neighbors_safe()
    print()

    test_neighbor_cache()
    print()

    test_correctness(N=512, C_in=16, C_out=32)
    print()
    test_correctness(N=1024, C_in=64, C_out=64)
    print()

    test_cache_amortization()
    print()

    # Real slat-hot-path-ish dimensions.
    bench(N=2048, C_in=64,  C_out=64)
    print()
    bench(N=4096, C_in=128, C_out=128)
    print()
    bench(N=4096, C_in=256, C_out=256)
    print()

    print("OK.")


if __name__ == "__main__":
    main()
