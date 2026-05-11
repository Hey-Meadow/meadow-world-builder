"""Mock-backbone tests for FlowMatching + CFGWrapper.

Run with the SAM 3D Body venv::

    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        meadow3d/tests/test_sampler.py
"""

from __future__ import annotations

import os
import sys

# Make the meadow3d package importable when run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx  # noqa: E402

from meadow3d.models.sampler_mlx import CFGWrapper, FlowMatching  # noqa: E402


# ---------------------------------------------------------------------------
# Mock backbones
# ---------------------------------------------------------------------------

class CountingBackbone:
    """Returns ``-x`` (cond-independent) and counts every call."""

    def __init__(self, return_value=None):
        self.calls = 0
        self.return_value = return_value  # if None, returns -x

    def __call__(self, x, t, cond):
        self.calls += 1
        if self.return_value is not None:
            return self.return_value
        return -x


class CondAwareBackbone:
    """Returns x*sign(cond.mean()) so cond branch != uncond branch."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, t, cond):
        self.calls += 1
        # cond branch -> +x, uncond (zeros) branch -> 0 -> v = 0.
        m = mx.mean(cond)
        # avoid division-by-zero edge: just return cond.mean() * x.
        return m * x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_cfg_shape_and_call_count():
    bb = CountingBackbone()
    sampler = FlowMatching(num_steps=10, cfg_strength=0.0)
    cond = mx.zeros((1, 4, 8))
    out = sampler.sample(bb, cond=cond, shape=(1, 4, 8))
    mx.eval(out)
    assert out.shape == (1, 4, 8), f"bad shape {out.shape}"
    assert bb.calls == 10, f"expected 10 backbone calls, got {bb.calls}"
    assert sampler.last_backbone_calls == 10
    print(f"[ok] no-CFG: shape={tuple(out.shape)} calls={bb.calls}")


def test_cfg_doubles_call_count():
    bb = CountingBackbone()
    sampler = FlowMatching(num_steps=8, cfg_strength=7.0)
    cond = mx.ones((1, 4, 8))
    out = sampler.sample(bb, cond=cond, shape=(1, 4, 8))
    mx.eval(out)
    assert out.shape == (1, 4, 8), f"bad shape {out.shape}"
    assert bb.calls == 16, f"expected 16 backbone calls (2*8), got {bb.calls}"
    assert sampler.last_backbone_calls == 16
    print(f"[ok] CFG=7: shape={tuple(out.shape)} calls={bb.calls}")


def test_cfg_strength_zero_skips_uncond():
    """Strength 0 must not invoke the uncond branch even via CFGWrapper."""
    bb = CountingBackbone()
    wrap = CFGWrapper(bb, cfg_strength=0.0)
    x = mx.ones((1, 2, 3))
    t = mx.array(0.5)
    cond = mx.ones((1, 2, 3))
    y = wrap(x, t, cond)
    mx.eval(y)
    assert bb.calls == 1, f"expected 1 call, got {bb.calls}"
    assert mx.allclose(y, -x).item()
    print(f"[ok] CFGWrapper strength=0: 1 call, identity to backbone")


def test_cfg_blend_formula():
    """Check (1+s)*v_cond - s*v_uncond exactly."""
    s = 7.0
    # backbone returns +1 for cond branch, 0 for uncond (cond is zeros).
    bb = CondAwareBackbone()
    wrap = CFGWrapper(bb, cfg_strength=s)
    x = mx.ones((1, 2, 3))
    cond = mx.ones((1, 2, 3))  # mean = 1 -> cond returns +x
    y = wrap(x, mx.array(0.5), cond)
    mx.eval(y)
    expected = (1.0 + s) * x - s * mx.zeros_like(x)
    assert mx.allclose(y, expected).item(), f"CFG blend mismatch"
    assert bb.calls == 2
    print(f"[ok] CFG blend: (1+{s})*v_c - {s}*v_u verified")


def test_time_schedule_linear():
    """t_seq must be linspace(0,1, N+1) with rescale=1, reversed=False."""
    sampler = FlowMatching(num_steps=4, cfg_strength=0.0)
    t = sampler._prepare_t()
    expected = mx.linspace(0.0, 1.0, 5)
    assert mx.allclose(t, expected).item(), f"got {t.tolist()}"
    print(f"[ok] linear schedule: {[round(float(v), 3) for v in t.tolist()]}")


def test_reversed_schedule():
    sampler = FlowMatching(num_steps=4, cfg_strength=0.0, reversed_timestamp=True)
    t = sampler._prepare_t()
    expected = 1.0 - mx.linspace(0.0, 1.0, 5)
    assert mx.allclose(t, expected).item()
    print(f"[ok] reversed schedule: {[round(float(v), 3) for v in t.tolist()]}")


def test_euler_integration_correctness():
    """With backbone v(x,t)=-x and dt small, ||x|| should decrease monotonically.

    This verifies the Euler step direction (x = x + v*dt) is correctly wired.
    """
    bb = CountingBackbone()  # returns -x
    sampler = FlowMatching(num_steps=20, cfg_strength=0.0, time_scale=1.0)
    x0 = mx.ones((1, 4))
    out = sampler.sample(bb, cond=mx.zeros((1, 4)), x0=x0)
    mx.eval(out)
    # Each step: x <- x + (-x) * dt = x*(1-dt). dt = 1/20 = 0.05.
    # After 20 steps: x = (0.95)^20 * 1 ~= 0.3585.
    expected_scalar = 0.95 ** 20
    expected = mx.full((1, 4), expected_scalar)
    diff = mx.max(mx.abs(out - expected)).item()
    assert diff < 1e-5, f"Euler integration drift: max|diff|={diff}"
    print(f"[ok] Euler v=-x: ||out||={mx.mean(out).item():.4f} expected={expected_scalar:.4f}")


def test_shape_with_3d_latent():
    """Realistic SAM 3D Objects latent shape: (B=1, N=tokens, C=channels)."""
    bb = CountingBackbone()
    sampler = FlowMatching(num_steps=25, cfg_strength=7.0)
    cond = mx.zeros((1, 256, 1024))
    out = sampler.sample(bb, cond=cond, shape=(1, 4096, 8))
    mx.eval(out)
    assert out.shape == (1, 4096, 8)
    assert bb.calls == 50  # 25 * 2 (CFG)
    print(f"[ok] realistic shape (1,4096,8): calls={bb.calls}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_no_cfg_shape_and_call_count,
        test_cfg_doubles_call_count,
        test_cfg_strength_zero_skips_uncond,
        test_cfg_blend_formula,
        test_time_schedule_linear,
        test_reversed_schedule,
        test_euler_integration_correctness,
        test_shape_with_3d_latent,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc!r}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
