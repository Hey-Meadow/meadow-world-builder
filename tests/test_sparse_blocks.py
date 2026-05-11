"""Smoke + load tests for ``meadow_wb.models.sparse_blocks_mlx``.

Exercises:

1. Random sparse input -> ``SparseConvBlock`` forward (no npz, init weights).
2. Load ``input_blocks`` and ``out_blocks`` from ``slat_flow.npz``; verify
   every key under their prefix is consumed.
3. Forward through the loaded ``SparseInputBlocks`` and ``SparseOutputBlocks``
   on a random sparse coord set; check shapes line up with the inferred
   channel progression.
4. Verify neighbor-table caching: the cache holds across repeated forwards
   that thread the *same* coords array (i.e. the cache is keyed on
   ``id(coords)``, so 50 ODE steps reuse one neighbor build per stage).

Run with:

    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        /Users/akaihuangm1/Desktop/github/sam-3d-objects/meadow_wb/tests/test_sparse_blocks.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = "/Users/akaihuangm1/Desktop/github/sam-3d-objects"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mlx.core as mx  # noqa: E402

from meadow_wb.kernels.sparse_subm_conv3d import (  # noqa: E402
    _NEIGHBOR_CACHE,
    clear_neighbor_cache,
)
from meadow_wb.models.sparse_blocks_mlx import (  # noqa: E402
    SparseConvBlock,
    SparseInputBlocks,
    SparseOutputBlocks,
)


_NPZ = os.path.join(
    _REPO, "meadow_wb", "weights", "sam3d_objects", "slat_flow.npz"
)
_PREFIX_INPUT = "reverse_fn.backbone.input_blocks."
_PREFIX_OUT = "reverse_fn.backbone.out_blocks."


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_random_voxels(N: int, grid: int = 32, seed: int = 0) -> mx.array:
    """N unique voxel coords in batch=0 within a `grid`^3 box."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    while len(out) < N:
        z, y, x = rng.integers(0, grid, size=3).tolist()
        if (z, y, x) in seen:
            continue
        seen.add((z, y, x))
        out.append((0, z, y, x))
    return mx.array(np.asarray(out, dtype=np.int32))


# ---------------------------------------------------------------------------
# 1) Smoke test: SparseConvBlock with random init
# ---------------------------------------------------------------------------


def test_smoke_sparseconvblock() -> None:
    print("[1] SparseConvBlock smoke (random init) ...", flush=True)
    N = 1000
    coords = make_random_voxels(N, grid=32)
    in_ch, out_ch = 32, 32
    emb_ch = 1024

    blk = SparseConvBlock(in_ch, out_ch, emb_channels=emb_ch)
    feats = mx.random.normal((N, in_ch))
    emb = mx.random.normal((1, emb_ch))

    out = blk(feats, coords, emb)
    assert out.shape == (N, out_ch), out.shape
    print(f"    OK   N={N}  in_ch={in_ch} -> out_ch={out_ch}, out.shape={out.shape}")


# ---------------------------------------------------------------------------
# 2) Load from npz; verify all keys consumed
# ---------------------------------------------------------------------------


def _flat_npz_dict(path: str) -> dict[str, mx.array]:
    raw = np.load(path)
    out: dict[str, mx.array] = {}
    for k in raw.files:
        out[k] = mx.array(raw[k])
    return out


def test_load_input_blocks() -> None:
    print(f"[2a] Load SparseInputBlocks from npz prefix {_PREFIX_INPUT!r} ...", flush=True)
    weights = _flat_npz_dict(_NPZ)
    keys_under = [k for k in weights if k.startswith(_PREFIX_INPUT)]
    print(f"    {len(keys_under)} keys under prefix")
    m = SparseInputBlocks.from_npz(weights, _PREFIX_INPUT)
    assert len(m._consumed_keys) == len(keys_under), (
        f"consumed={len(m._consumed_keys)}  total={len(keys_under)}"
    )
    print(f"    OK   {len(m.blocks)} blocks, all {len(keys_under)} keys consumed")
    for i, blk in enumerate(m.blocks):
        print(
            f"          block {i}: in={blk.in_channels:5d} out={blk.out_channels:5d} "
            f"downsample={blk.downsample}  has_skip_linear={blk.has_skip_linear}"
        )
    return m, weights


def test_load_out_blocks() -> None:
    print(f"[2b] Load SparseOutputBlocks from npz prefix {_PREFIX_OUT!r} ...", flush=True)
    weights = _flat_npz_dict(_NPZ)
    keys_under = [k for k in weights if k.startswith(_PREFIX_OUT)]
    print(f"    {len(keys_under)} keys under prefix")
    m = SparseOutputBlocks.from_npz(weights, _PREFIX_OUT)
    assert len(m._consumed_keys) == len(keys_under), (
        f"consumed={len(m._consumed_keys)}  total={len(keys_under)}"
    )
    print(f"    OK   {len(m.blocks)} blocks, all {len(keys_under)} keys consumed")
    for i, blk in enumerate(m.blocks):
        print(
            f"          block {i}: in={blk.in_channels:5d} out={blk.out_channels:5d} "
            f"upsample={blk.upsample}  has_skip_linear={blk.has_skip_linear}"
        )
    return m, weights


# ---------------------------------------------------------------------------
# 3) Forward shape sanity, with the loaded weights
# ---------------------------------------------------------------------------


def test_forward_shapes(input_blocks: SparseInputBlocks,
                        out_blocks: SparseOutputBlocks) -> None:
    print("[3] Forward through loaded blocks (shape check) ...", flush=True)

    # Use small N so this stays cheap. coords are stable across the call --
    # what we want is that input_blocks.0 maps (N, 128) -> (N, 128) and
    # input_blocks.1 maps (N, 128) -> (N, 1024) with the SAME coord set
    # (we test the post-downsample-coords path separately by re-using the
    # same coord array; SparseDownsample is the caller's responsibility).
    N = 256
    coords = make_random_voxels(N, grid=32)
    emb = mx.random.normal((1, 1024))

    in_ch_0 = input_blocks.blocks[0].in_channels  # 128
    feats = mx.random.normal((N, in_ch_0))

    skips, h, c = input_blocks(feats, coords, emb)
    assert h.shape == (N, input_blocks.blocks[-1].out_channels), h.shape
    print(f"    input  : N={N}  feats {in_ch_0} -> {h.shape[-1]}, "
          f"{len(skips)} skips with channels "
          f"{[s.shape[-1] for s in skips]}")

    # The transformer middle-stack would update h here; we just feed the
    # input output directly back into out_blocks (with skip concat the
    # channels match: 1024 + skip[1024] = 2048 -> first out block).
    h2, c2 = out_blocks(h, c, emb, skips, use_skip_connection=True)
    assert h2.shape == (N, out_blocks.blocks[-1].out_channels), h2.shape
    print(f"    output : feats {h.shape[-1]} -> {h2.shape[-1]} (final)")


# ---------------------------------------------------------------------------
# 4) Neighbor-table caching: same coords -> single build across many calls
# ---------------------------------------------------------------------------


def test_neighbor_cache(input_blocks: SparseInputBlocks) -> None:
    print("[4] Neighbor-table cache reuse across simulated ODE steps ...", flush=True)
    clear_neighbor_cache()
    N = 256
    coords = make_random_voxels(N, grid=32)        # ONE array, threaded 50x
    emb = mx.random.normal((1, 1024))
    feats = mx.random.normal((N, input_blocks.blocks[0].in_channels))

    # First pass: cache cold.
    _ = input_blocks(feats, coords, emb)
    mx.eval(_[1])
    builds_after_first = len(_NEIGHBOR_CACHE)

    # 49 more forward passes with the SAME coords array.
    for _i in range(49):
        out = input_blocks(feats, coords, emb)
        mx.eval(out[1])
    builds_after_50 = len(_NEIGHBOR_CACHE)

    print(f"    cache size after 1 forward  : {builds_after_first}")
    print(f"    cache size after 50 forwards: {builds_after_50}")
    assert builds_after_first == builds_after_50, (
        "Neighbor cache grew across repeated calls -- caching is broken."
    )
    # We expect at most ONE entry (one coords array reused by both
    # input_blocks.0 and input_blocks.1 since neither updown actually mutates
    # the array passed in this test; each conv1/conv2 within a block hits the
    # same coords id, so a single build serves all 4 conv calls per forward).
    assert builds_after_50 == 1, (
        f"Expected 1 cached neighbor table (single coords array), got {builds_after_50}"
    )
    print("    OK   neighbor table built once, reused for all 50 ODE steps")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    test_smoke_sparseconvblock()

    if not os.path.exists(_NPZ):
        print(f"[skip] npz weights not found at {_NPZ}; remaining tests skipped.")
        return 0

    input_blocks, _ = test_load_input_blocks()
    out_blocks, _ = test_load_out_blocks()
    test_forward_shapes(input_blocks, out_blocks)
    test_neighbor_cache(input_blocks)
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
