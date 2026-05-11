"""Inference-time caching utilities for diffusion / flow-matching samplers.

Implements the temporal half of Fast-SAM3D mechanism ②: curvature-aware
tangent reuse. The sampler runs a full backbone evaluation at every step
unless the cache says "this step is in a quasi-linear regime, reuse the
tangent from the last anchor". When accumulated relative error exceeds
``error_threshold``, force a full evaluation and refresh the anchor.

See ``docs/SPEC_FAST_SAM3D.md`` §2 for the math.

Spatial token carving (the other half of mechanism ②) lives in
``saliency.py``; this file is curvature/tangent only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class CurvatureCacheStats:
    full_evals: int = 0
    cache_hits: int = 0
    last_E: float = 0.0
    last_kappa: float = 0.0

    @property
    def total(self) -> int:
        return self.full_evals + self.cache_hits

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.total == 0 else self.cache_hits / self.total


class CurvatureCache:
    """Tangent-reuse cache for flow-matching Euler integration.

    Algorithm (per step, given current state ``x_t``):

    1. **Warmup** — for the first ``warmup`` steps, always run the full
       backbone and accumulate two anchors so curvature ``kappa`` can be
       estimated.
    2. **Predict** — once we have an anchor ``(x_a, v_a)`` and a
       curvature estimate ``kappa``, predict the relative error from
       reusing the tangent ``delta = v_a - x_a`` at step ``t`` as
       ``eps_t ≈ kappa * ||x_t - x_{t-1}|| / ||v_{t-1}||``.
    3. **Switch** — accumulate ``E_t = sum(eps_n)`` since the last
       anchor. If ``E_t < error_threshold``, return cached tangent.
       Otherwise force a full eval and refresh the anchor.

    The cache is stateful — instantiate once per sampler run, then call
    :meth:`maybe_skip` inside the integration loop.
    """

    def __init__(
        self,
        error_threshold: float = 1.5,
        warmup: int = 2,
    ) -> None:
        self.error_threshold = float(error_threshold)
        self.warmup = int(warmup)
        self.reset()

    def reset(self) -> None:
        self._anchor_x: Optional[mx.array] = None
        self._anchor_v: Optional[mx.array] = None
        self._prev_x: Optional[mx.array] = None
        self._prev_v: Optional[mx.array] = None
        self._kappa: float = 0.0
        self._kappa_measured: bool = False  # distinguishes 0.0 from unmeasured
        self._E_acc: float = 0.0
        self._step: int = 0
        self.stats = CurvatureCacheStats()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_norm(a: mx.array) -> float:
        """L2 norm flattened across all axes, returned as a float."""
        return float(mx.linalg.norm(a.reshape(-1)).item())

    def _predict_eps(self, x_t: mx.array) -> float:
        """Estimate per-step relative error if we reuse the tangent."""
        if (
            self._prev_v is None
            or self._prev_x is None
            or not self._kappa_measured
        ):
            return float("inf")
        dx = self._flat_norm(x_t - self._prev_x)
        v_norm = self._flat_norm(self._prev_v)
        if v_norm < 1e-8:
            return float("inf")
        return self._kappa * dx / v_norm

    def _refresh_anchor(self, x_t: mx.array, v_t: mx.array) -> None:
        # Update curvature kappa from two most recent full evaluations.
        if self._anchor_x is not None and self._anchor_v is not None:
            dv = self._flat_norm(v_t - self._anchor_v)
            dx = self._flat_norm(x_t - self._anchor_x)
            self._kappa = dv / max(dx, 1e-8)
            self._kappa_measured = True
        self._anchor_x = x_t
        self._anchor_v = v_t
        self._E_acc = 0.0

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------

    def maybe_skip(
        self,
        x_t: mx.array,
        full_eval_fn,
    ) -> mx.array:
        """Return the velocity ``v_t`` for the current step.

        Either runs ``full_eval_fn(x_t)`` (always returning the true
        backbone output) and refreshes the anchor, or — when the
        accumulated error budget is healthy — returns the cached
        tangent ``x_t + (v_anchor - x_anchor)`` without ever calling the
        backbone.
        """
        force_full = (
            self._step < self.warmup
            or self._anchor_v is None
            or self._anchor_x is None
        )

        if not force_full:
            eps = self._predict_eps(x_t)
            E_next = self._E_acc + eps
            if E_next >= self.error_threshold:
                force_full = True

        if force_full:
            v_t = full_eval_fn(x_t)
            self._refresh_anchor(x_t, v_t)
            self.stats.full_evals += 1
        else:
            # tangent reuse: v_t = x_t + (v_a - x_a)
            delta = self._anchor_v - self._anchor_x
            v_t = x_t + delta
            self._E_acc = E_next  # type: ignore[possibly-undefined]
            self.stats.cache_hits += 1

        self._prev_x = x_t
        self._prev_v = v_t
        self._step += 1
        self.stats.last_E = self._E_acc
        self.stats.last_kappa = self._kappa
        return v_t


__all__ = ["CurvatureCache", "CurvatureCacheStats"]
