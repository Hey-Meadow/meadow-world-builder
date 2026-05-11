"""Submanifold sparse 3D convolution (SubMConv3d) — MLX port.

Replaces ``spconv.pytorch.SubMConv3d`` (the dominant CUDA hot-path in the
SAM 3D Objects slat generator) with two pure-Apple-Silicon paths:

1. **Pure MLX path** — ``mx.take`` + ``mx.einsum``. Always used as the
   correctness oracle and as the default forward (Metal does not beat it
   for the hyper-parameter regime we hit; see report at the bottom of
   ``test_sparse_subm_conv3d.py``).
2. **Custom Metal kernel** — one threadgroup per output voxel, gather 27
   neighbors and run a 27 × C_in × C_out fused matmul. Available via the
   ``backend="metal"`` constructor argument; falls back to pure-MLX on
   any compile error.

Submanifold semantics
---------------------
SubMConv3d preserves coords: ``out_coords == in_coords`` for every voxel.
Each voxel ``v`` at ``(b, z, y, x)`` aggregates contributions from the
(up to 27) input voxels at offsets ``(b, z+dz, y+dy, x+dx)`` for
``(dz, dy, dx) in [-1,0,1]^3`` *that exist in the input set*.

Because input and output coord sets are identical, the **(N, 27)
neighbor lookup table** depends only on ``coords`` and is reused across
all ~800 SubMConv3d calls per inference. We cache it by
``id(coords)`` (the same MLX array is threaded through the slat-gen ODE
loop, so the id is stable and cheap to key on).

Spconv weight layout convention: ``(KD, KH, KW, C_in, C_out)`` =
``(3, 3, 3, C_in, C_out)``. We flatten ``(KD, KH, KW)`` to a single
27-axis with row-major order ``k = (dz+1)*9 + (dy+1)*3 + (dx+1)``.

References
----------
- SPEC: ``meadow3d/docs/SPEC_METAL_SPARSE.md``
- CUDA recon: ``meadow3d/docs/CUDA_DEPS.md``
- Source code: ``sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_spconv.py``
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Neighbor table (built on CPU with numpy; result lives on the MLX device).
# ---------------------------------------------------------------------------

# Module-level cache: id(coords) -> (coords_shape, kernel_size, neighbor_table_mxarray).
# Keyed on Python id so the *same* coords array threaded through the ODE loop
# only triggers one build. We also store the shape to defend against the rare
# case of id reuse across freed arrays.
_NEIGHBOR_CACHE: dict[int, tuple[tuple, int, mx.array]] = {}


def _coord_key(b: int, z: int, y: int, x: int) -> int:
    """Pack a (b, z, y, x) tuple into a single int. Coord components fit in
    16-bit signed (slat grid is at most 64^3 with batch < 16, well within
    signed 16-bit each), so a 64-bit key is plenty."""
    return ((int(b) & 0xFFFF) << 48) | ((int(z) & 0xFFFF) << 32) | ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)


def build_neighbor_table(coords: mx.array, kernel_size: int = 3) -> mx.array:
    """Compute (N, k**3) int32 neighbor table for submanifold convolution.

    Args:
        coords: (N, 4) int32 MLX array of (batch, z, y, x).
        kernel_size: kernel side length, must be odd. Default 3 → 27 offsets.

    Returns:
        (N, kernel_size**3) int32 MLX array. Entry ``[n, k]`` is the row
        index of the neighbor at offset ``k`` from voxel ``n``, or ``-1``
        if no such voxel exists in the input set.

    Result is cached on ``id(coords)`` so repeat calls are O(1).
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    cache_key = id(coords)
    cached = _NEIGHBOR_CACHE.get(cache_key)
    if cached is not None:
        cshape, ck, nt = cached
        if cshape == tuple(coords.shape) and ck == kernel_size:
            return nt

    # CPU build: hash dict, then assemble (N, K3).
    coords_np = np.asarray(coords)
    if coords_np.dtype != np.int32:
        coords_np = coords_np.astype(np.int32)
    assert coords_np.ndim == 2 and coords_np.shape[1] == 4, (
        f"coords must be (N, 4); got {coords_np.shape}"
    )

    N = coords_np.shape[0]
    K = kernel_size
    K3 = K * K * K
    half = K // 2

    # Build coord -> row hash.
    coord2row: dict[int, int] = {}
    for n in range(N):
        b, z, y, x = coords_np[n]
        coord2row[_coord_key(b, z, y, x)] = n

    nt = np.full((N, K3), -1, dtype=np.int32)
    # Offset order: k = (dz+half)*K*K + (dy+half)*K + (dx+half).
    # This matches the row-major flattening of weight (K, K, K, ...) when
    # we later index weight[dz, dy, dx, ...].
    for n in range(N):
        b, z, y, x = coords_np[n]
        for dz in range(-half, half + 1):
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    k = (dz + half) * K * K + (dy + half) * K + (dx + half)
                    key = _coord_key(b, z + dz, y + dy, x + dx)
                    r = coord2row.get(key, -1)
                    nt[n, k] = r

    nt_mx = mx.array(nt)
    _NEIGHBOR_CACHE[cache_key] = (tuple(coords.shape), kernel_size, nt_mx)
    return nt_mx


def clear_neighbor_cache() -> None:
    """Reset the neighbor table cache (call between inferences)."""
    _NEIGHBOR_CACHE.clear()


# ---------------------------------------------------------------------------
# Pure-MLX forward (oracle + default fast path).
# ---------------------------------------------------------------------------


def submconv3d_mlx(
    features: mx.array,
    neighbor_table: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
) -> mx.array:
    """Pure-MLX SubMConv3d forward via gather + einsum.

    Args:
        features: (N, C_in) MLX array.
        neighbor_table: (N, K3) int32 MLX array; -1 for absent neighbors.
        weight: (K3, C_in, C_out) MLX array.
        bias: (C_out,) MLX array or None.

    Returns:
        (N, C_out) MLX array.

    Implementation:
        1. Append a row of zeros to ``features`` so ``-1`` indexing
           (after remapping) becomes a no-op gather.
        2. Gather ``features_padded[neighbor_table_remap]`` to (N, K3, C_in).
        3. ``einsum('nkc,kco->no', neighbors, weight)``.
    """
    N, C_in = features.shape
    K3, _, C_out = weight.shape

    # Remap -1 -> N (last row of padded features, which is zeros).
    nt = mx.where(neighbor_table < 0, N, neighbor_table)

    # Pad features with a zero row at index N.
    zero_row = mx.zeros((1, C_in), dtype=features.dtype)
    feats_padded = mx.concatenate([features, zero_row], axis=0)  # (N+1, C_in)

    # Gather: (N, K3, C_in).
    neighbors = feats_padded[nt]  # MLX fancy-indexing with int32 is supported.

    # Contract: (N, K3, C_in) x (K3, C_in, C_out) -> (N, C_out).
    out = mx.einsum("nkc,kco->no", neighbors, weight)

    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Custom Metal kernel — gather + fused matmul, one threadgroup per output voxel.
# ---------------------------------------------------------------------------


_METAL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# One threadgroup per output voxel.
# THREADS threads cooperate to compute the C_OUT-vector for this voxel.
# Each thread takes a stride of THREADS through the C_out dimension.
# Per (k, c_in) pair: every thread reads features[neigh, c_in] (broadcast
# via threadgroup memory) and weight[k, c_in, my_c_out_chunk].
_METAL_SOURCE = """
    constexpr uint THREADS = THREADS_TPL;
    constexpr uint K3      = K3_TPL;

    uint voxel = threadgroup_position_in_grid.x;
    uint tid   = thread_position_in_threadgroup.x;
    uint N     = N_buf[0];
    uint Cin   = Cin_buf[0];
    uint Cout  = Cout_buf[0];

    if (voxel >= N) return;

    threadgroup float feat_share;

    // Output accumulators: one per (c_out chunk) handled by this thread.
    // We use stride-THREADS chunking, so up to ceil(Cout/THREADS) = MAX_OUT_PER_THREAD
    // values per thread. We dynamically loop and write directly.
    // Simpler: do one c_out value at a time.
    for (uint co = tid; co < Cout; co += THREADS) {
        float acc = 0.0f;
        for (uint k = 0; k < K3; ++k) {
            int neigh = neighbor_table[voxel * K3 + k];
            if (neigh < 0) continue;
            uint un = (uint)neigh;
            // Loop over Cin contribution.
            // weight layout: (K3, Cin, Cout) row-major.
            //   weight[k, ci, co] = weight[(k*Cin + ci)*Cout + co]
            // features[un, ci] = features[un*Cin + ci]
            float local_sum = 0.0f;
            for (uint ci = 0; ci < Cin; ++ci) {
                float f = features[un * Cin + ci];
                float w = weight[(k * Cin + ci) * Cout + co];
                local_sum = fma(f, w, local_sum);
            }
            acc += local_sum;
        }
        if (HAS_BIAS_TPL) {
            acc += bias[co];
        }
        out[voxel * Cout + co] = acc;
    }
"""


_METAL_KERNEL_CACHE: dict[tuple[int, int, bool], object] = {}


def _get_metal_kernel(threads: int, k3: int, has_bias: bool):
    key = (threads, k3, has_bias)
    k = _METAL_KERNEL_CACHE.get(key)
    if k is not None:
        return k
    src = (
        _METAL_SOURCE
        .replace("THREADS_TPL", str(threads))
        .replace("K3_TPL", str(k3))
        .replace("HAS_BIAS_TPL", "1" if has_bias else "0")
    )
    inputs = ["features", "neighbor_table", "weight", "N_buf", "Cin_buf", "Cout_buf"]
    if has_bias:
        inputs.append("bias")
    k = mx.fast.metal_kernel(
        name=f"submconv3d_{threads}_{k3}_{int(has_bias)}",
        input_names=inputs,
        output_names=["out"],
        header=_METAL_HEADER,
        source=src,
    )
    _METAL_KERNEL_CACHE[key] = k
    return k


def submconv3d_metal(
    features: mx.array,
    neighbor_table: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    threads: int = 64,
) -> mx.array:
    """Custom Metal kernel forward. See ``submconv3d_mlx`` for semantics.

    Designed for small C_in/C_out (the slat hot-path uses 64-768 channels).
    """
    N, C_in = features.shape
    K3, _, C_out = weight.shape

    has_bias = bias is not None
    kernel = _get_metal_kernel(threads, K3, has_bias)

    # Scalar buffers (MLX kernels can't take Python ints — pass as 1-elem arrays).
    N_buf   = mx.array([N], dtype=mx.uint32)
    Cin_buf = mx.array([C_in], dtype=mx.uint32)
    Cout_buf = mx.array([C_out], dtype=mx.uint32)

    inputs = [features, neighbor_table.astype(mx.int32), weight, N_buf, Cin_buf, Cout_buf]
    if has_bias:
        inputs.append(bias)

    outs = kernel(
        inputs=inputs,
        grid=(N * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(N, C_out)],
        output_dtypes=[features.dtype],
    )
    return outs[0]


# ---------------------------------------------------------------------------
# High-level module — what callers should use.
# ---------------------------------------------------------------------------


class SparseSubmConv3d:
    """Submanifold sparse 3D convolution with cached neighbor table.

    Drop-in semantic replacement for ``spconv.pytorch.SubMConv3d`` (kernel
    3, stride 1, no padding) when both input and output share the same
    coord set.

    Weight convention (matches spconv): ``(KD, KH, KW, C_in, C_out)``.
    We store the flattened view ``(K**3, C_in, C_out)``.

    Args:
        in_channels: C_in.
        out_channels: C_out.
        kernel_size: 3 (only kernel size 3 is exercised in SAM 3D Objects).
        bias: include bias term.
        backend: ``"mlx"`` (gather + einsum, default) or ``"metal"``
            (custom Metal kernel — for benchmarking; falls back to mlx
            if compilation fails at first call).

    Initialization:
        Weights are zero-initialized. Use ``load_weight()`` to install
        a converted spconv weight tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        backend: str = "mlx",
    ):
        assert kernel_size == 3, "Only kernel_size=3 supported (matches all spconv call sites in SAM 3D Objects)."
        assert backend in ("mlx", "metal")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K3 = kernel_size ** 3
        self.backend = backend

        # Default zero-init; real weights come from .load_weight().
        self.weight = mx.zeros((self.K3, in_channels, out_channels), dtype=mx.float32)
        self.bias: Optional[mx.array] = (
            mx.zeros((out_channels,), dtype=mx.float32) if bias else None
        )

    # -- weight loaders ------------------------------------------------------

    def load_weight(self, weight: mx.array, bias: Optional[mx.array] = None) -> None:
        """Install weights from a converted checkpoint.

        Args:
            weight: either ``(K, K, K, C_in, C_out)`` (spconv layout) or
                already-flattened ``(K**3, C_in, C_out)``.
            bias: ``(C_out,)`` or None.
        """
        if weight.ndim == 5:
            K = self.kernel_size
            assert weight.shape == (K, K, K, self.in_channels, self.out_channels), (
                f"weight shape {weight.shape} != ({K},{K},{K},{self.in_channels},{self.out_channels})"
            )
            weight = weight.reshape(self.K3, self.in_channels, self.out_channels)
        else:
            assert weight.shape == (self.K3, self.in_channels, self.out_channels), (
                f"weight shape {weight.shape} != ({self.K3},{self.in_channels},{self.out_channels})"
            )
        self.weight = weight
        if bias is not None:
            assert bias.shape == (self.out_channels,)
            self.bias = bias

    # -- forward -------------------------------------------------------------

    def __call__(self, features: mx.array, coords: mx.array) -> mx.array:
        """Forward pass.

        Args:
            features: (N, C_in) MLX array.
            coords: (N, 4) int32 MLX array of (batch, z, y, x).

        Returns:
            (N, C_out) MLX array. Output coords == input coords (submanifold).
        """
        nt = build_neighbor_table(coords, self.kernel_size)
        if self.backend == "metal":
            try:
                return submconv3d_metal(features, nt, self.weight, self.bias)
            except Exception:
                # Fall back to MLX path on first-call compile error.
                self.backend = "mlx"
        return submconv3d_mlx(features, nt, self.weight, self.bias)


__all__ = [
    "SparseSubmConv3d",
    "build_neighbor_table",
    "clear_neighbor_cache",
    "submconv3d_mlx",
    "submconv3d_metal",
]
