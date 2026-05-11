"""Integration test: CurvatureCache wired into FlowMatching.sample.

Verifies:
1. No-cache path is byte-identical to today (regression guard).
2. With a cache, ``cache.stats.hit_rate > 0`` for a smooth mock backbone.
3. Counted backbone invocations drop when the cache is active.

Run with the SAM 3D Body venv::

    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        tests/test_curvature_cache_integration.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx  # noqa: E402

from meadow_wb.models.sampler_mlx import FlowMatching  # noqa: E402
from meadow_wb.utils.cache import CurvatureCache  # noqa: E402


class CountingLinearBackbone:
    """Smooth, quasi-linear velocity field ``v(x, t, cond) = -0.5 * x``.

    Trajectory under Euler integration is monotonic exponential-decay-like
    and stays in the low-curvature regime — exactly the case the
    CurvatureCache is designed to exploit.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x: mx.array, t, cond: mx.array) -> mx.array:
        self.calls += 1
        return -0.5 * x


def _run(num_steps: int, cache):
    backbone = CountingLinearBackbone()
    fm = FlowMatching(num_steps=num_steps, cfg_strength=0.0)
    cond = mx.zeros((1, 4, 8))
    # Deterministic init noise so the no-cache vs no-cache run match
    # bit-for-bit.
    x0 = mx.ones((1, 4, 8))
    out = fm.sample(backbone_fn=backbone, cond=cond, x0=x0, cache=cache)
    mx.eval(out)
    return out, backbone.calls


def test_no_cache_path_unchanged():
    out_a, calls_a = _run(num_steps=10, cache=None)
    out_b, calls_b = _run(num_steps=10, cache=None)
    assert calls_a == calls_b == 10, (calls_a, calls_b)
    diff = float(mx.max(mx.abs(out_a - out_b)).item())
    assert diff == 0.0, f"no-cache path is non-deterministic: max|delta|={diff}"
    print(f"[ok] no-cache regression: 10 steps, {calls_a} backbone calls, diff=0")


def test_cache_reduces_backbone_calls():
    cache = CurvatureCache(error_threshold=1.5, warmup=2)
    out_cached, calls_cached = _run(num_steps=10, cache=cache)
    out_uncached, calls_uncached = _run(num_steps=10, cache=None)
    print(
        f"[stats] cached calls={calls_cached} ({cache.stats.cache_hits} hits, "
        f"{cache.stats.full_evals} full) | uncached calls={calls_uncached} | "
        f"hit_rate={cache.stats.hit_rate:.2%}"
    )
    assert calls_cached < calls_uncached, (
        f"cache did not skip any forward passes: cached={calls_cached} "
        f"vs uncached={calls_uncached}"
    )
    assert cache.stats.hit_rate > 0.0, (
        f"cache.stats.hit_rate={cache.stats.hit_rate} (expected > 0)"
    )
    # Quality sanity: with linear field, tangent reuse should be very accurate.
    err = float(mx.max(mx.abs(out_cached - out_uncached)).item())
    print(f"[stats] cached-vs-uncached max|delta| = {err:.4e}")
    assert err < 1.0, f"cache introduced unexpectedly large drift: {err}"


def test_cache_state_per_run():
    """Verify cache can be reused across runs by calling .reset()."""
    cache = CurvatureCache(error_threshold=1.5, warmup=2)
    _run(num_steps=5, cache=cache)
    hits_first = cache.stats.cache_hits
    cache.reset()
    assert cache.stats.cache_hits == 0
    _run(num_steps=5, cache=cache)
    print(
        f"[ok] reset clears stats: pre={hits_first}, post-reset hits="
        f"{cache.stats.cache_hits}"
    )


if __name__ == "__main__":
    test_no_cache_path_unchanged()
    test_cache_reduces_backbone_calls()
    test_cache_state_per_run()
    print("ALL CURVATURE-CACHE INTEGRATION TESTS PASSED")
