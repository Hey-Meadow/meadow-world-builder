"""Flow matching ODE sampler + Classifier-Free Guidance wrapper (MLX).

Pure MLX port of:
  sam3d_objects/model/backbone/generator/flow_matching/{model.py,solver.py}
  sam3d_objects/model/backbone/generator/classifier_free_guidance.py

This module has NO learned weights - it is pure ODE arithmetic. The DiT
backbone is plugged in via a ``backbone_fn`` callable.

Conventions
-----------
PT source integrates from t=0 (noise x_0) to t=1 (clean x_1) using a
``linspace(0, 1, steps+1)`` schedule (rectifier flow, sigma_min=0). The
forward-Euler step is therefore::

    x_{t+dt} = x_t + v(x_t, t) * dt

The backbone receives ``t * time_scale`` (PT default ``time_scale=1000``).

Classifier-free guidance (per PT ``ClassifierFreeGuidance``)::

    v_cfg = (1 + s) * v_cond - s * v_uncond
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import mlx.core as mx


# ---------------------------------------------------------------------------
# CFG wrapper
# ---------------------------------------------------------------------------

class CFGWrapper:
    """Classifier-Free Guidance wrapper around a backbone forward.

    Parameters
    ----------
    backbone : Callable
        Signature ``(x, t, cond) -> velocity``. ``cond`` may be any pytree
        of mlx arrays / non-tensor metadata; the wrapper only swaps it
        between conditional / unconditional branches.
    cfg_strength : float
        CFG strength ``s``. ``s == 0`` disables CFG (single backbone call).
    uncond_token : mx.array | None
        Replacement conditioning used for the unconditional branch. If
        ``None`` we follow PT's ``unconditional_handling="zeros"`` and pass
        ``mx.zeros_like(cond)``.
    interval : tuple[float, float] | None
        If given, CFG is only applied when ``interval[0] <= t <= interval[1]``;
        outside the interval we fall back to the conditional branch only
        (matches PT ``get_strength``).
    """

    def __init__(
        self,
        backbone: Callable[[mx.array, mx.array, mx.array], mx.array],
        cfg_strength: float = 7.0,
        uncond_token: Optional[mx.array] = None,
        interval: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.backbone = backbone
        self.cfg_strength = float(cfg_strength)
        self.uncond_token = uncond_token
        self.interval = interval
        # Diagnostic: counts how many times the underlying backbone is invoked.
        self.call_count = 0

    # -- helpers ---------------------------------------------------------
    def _strength_at(self, t_value: float) -> float:
        if self.cfg_strength <= 0.0:
            return 0.0
        if self.interval is None:
            return self.cfg_strength
        lo, hi = self.interval
        return self.cfg_strength if (lo <= t_value <= hi) else 0.0

    def _make_uncond(self, cond: mx.array) -> mx.array:
        if self.uncond_token is not None:
            return self.uncond_token
        # PT default is unconditional_handling="zeros" -> torch.zeros_like
        return mx.zeros_like(cond)

    # -- forward ---------------------------------------------------------
    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        # Resolve scalar t for interval test (t may be 0-d array).
        t_value = float(t.item()) if hasattr(t, "item") else float(t)
        s = self._strength_at(t_value)

        v_cond = self.backbone(x, t, cond)
        self.call_count += 1
        if s == 0.0:
            return v_cond

        uncond = self._make_uncond(cond)
        v_uncond = self.backbone(x, t, uncond)
        self.call_count += 1
        # PT: y = (1 + s) * y_cond - s * y_uncond
        return (1.0 + s) * v_cond - s * v_uncond


# ---------------------------------------------------------------------------
# Flow matching Euler sampler
# ---------------------------------------------------------------------------

class FlowMatching:
    """Euler-method flow-matching sampler in MLX.

    Parameters
    ----------
    num_steps : int
        Number of Euler steps. PT default is 100; per RECON_INFER.md /
        SPEC_SAMPLER.md the deployed inference uses ~25.
    sigma_max : float
        Std-dev of the initial noise sample. PT samples ``randn`` (sigma=1).
    cfg_strength : float
        Default CFG strength used when ``backbone_fn`` is wrapped here.
        7.0 for shape, 5.0 for texture per RECON_INFER.md. ``0.0``
        disables CFG (single backbone call per step).
    schedule : str
        ``"linear"`` -> ``linspace(0, 1, num_steps+1)`` (PT default).
    time_scale : float
        Scale applied to ``t`` before passing to the backbone. PT default 1000.
    rescale_t : float
        PT ``rescale_t`` parameter; ``1.0`` is identity. The transform is
        ``t / (1 + (rescale_t - 1) * (1 - t))``.
    reversed_timestamp : bool
        If True the schedule is flipped to ``1 - t`` before iterating
        (matches PT ``reversed_timestamp``).
    """

    def __init__(
        self,
        num_steps: int = 25,
        sigma_max: float = 1.0,
        cfg_strength: float = 7.0,
        schedule: str = "linear",
        time_scale: float = 1000.0,
        rescale_t: float = 1.0,
        reversed_timestamp: bool = False,
    ) -> None:
        if schedule != "linear":
            raise ValueError(
                f"only 'linear' schedule is implemented (got {schedule!r}); "
                "PT source uses linspace-based linear schedule"
            )
        self.num_steps = int(num_steps)
        self.sigma_max = float(sigma_max)
        self.cfg_strength = float(cfg_strength)
        self.schedule = schedule
        self.time_scale = float(time_scale)
        self.rescale_t = float(rescale_t)
        self.reversed_timestamp = bool(reversed_timestamp)

    # -- time grid -------------------------------------------------------
    def _prepare_t(self) -> mx.array:
        """Replicates ``FlowMatching._prepare_t`` from PT source."""
        t_seq = mx.linspace(0.0, 1.0, self.num_steps + 1)
        if self.rescale_t != 1.0:
            t_seq = t_seq / (1.0 + (self.rescale_t - 1.0) * (1.0 - t_seq))
        if self.reversed_timestamp:
            t_seq = 1.0 - t_seq
        return t_seq

    # -- noise -----------------------------------------------------------
    def _generate_noise(
        self, shape: Tuple[int, ...], key: Optional[mx.array]
    ) -> mx.array:
        if key is not None:
            x = mx.random.normal(shape=shape, key=key)
        else:
            x = mx.random.normal(shape=shape)
        if self.sigma_max != 1.0:
            x = x * self.sigma_max
        return x

    # -- sample ----------------------------------------------------------
    def sample(
        self,
        backbone_fn: Callable[[mx.array, mx.array, mx.array], mx.array],
        cond: mx.array,
        uncond: Optional[mx.array] = None,
        shape: Optional[Tuple[int, ...]] = None,
        positions: Optional[mx.array] = None,  # noqa: ARG002 (reserved)
        cfg_strength: Optional[float] = None,
        key: Optional[mx.array] = None,
        x0: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        """Run Euler ODE integration and return the final ``x_1`` latent.

        Parameters
        ----------
        backbone_fn
            Callable ``(x, t_scaled, cond) -> velocity``. ``t_scaled`` is
            already multiplied by ``time_scale``; the callable does not
            need to re-scale.
        cond
            Conditioning array (e.g. image tokens).
        uncond
            Optional pre-computed unconditional conditioning. If not given
            and CFG is active we use ``zeros_like(cond)`` (PT default).
        shape
            Latent shape ``(B, N, C)``. Required if ``x0`` is None.
        positions
            Optional positional embeddings - reserved, currently unused at
            the sampler level (the backbone owns positional encoding).
        cfg_strength
            Override the constructor default.
        key
            Optional PRNG key for ``mx.random.normal``.
        x0
            Optional initial noise tensor (overrides ``shape``-based draw).
        cache
            Optional :class:`~meadow_wb.utils.cache.CurvatureCache` instance.
            When provided, every ``step_fn`` call inside the Euler loop is
            routed through ``cache.maybe_skip`` so quasi-linear segments can
            reuse the previous tangent instead of forwarding the backbone.
            When ``None`` (default) the loop is byte-identical to before.

        Returns
        -------
        mx.array
            Final ``x_1`` (denoised) latent.
        """
        s = self.cfg_strength if cfg_strength is None else float(cfg_strength)

        # Wrap backbone with CFG when strength > 0; otherwise pass-through.
        if s > 0.0:
            wrapped = CFGWrapper(
                backbone_fn,
                cfg_strength=s,
                uncond_token=uncond,
            )
            step_fn: Callable[[mx.array, mx.array, mx.array], mx.array] = wrapped
        else:
            wrapped = None
            step_fn = backbone_fn  # type: ignore[assignment]

        # Initial noise x_0 ~ N(0, sigma_max^2 I).
        if x0 is None:
            if shape is None:
                raise ValueError("either `shape` or `x0` must be provided")
            x = self._generate_noise(shape, key)
        else:
            x = x0

        t_seq = self._prepare_t()

        # Euler integration: x_{t+dt} = x + v(x, t) * dt.
        for i in range(self.num_steps):
            t0 = t_seq[i]
            t1 = t_seq[i + 1]
            dt = t1 - t0
            t_in = t0 * self.time_scale
            if cache is None:
                v = step_fn(x, t_in, cond)
            else:
                # Bind t_in/cond into a unary closure so the cache only
                # sees the state x_t — matches the spec's full_eval_fn(x_t)
                # signature and lets the no-cache path stay untouched.
                v = cache.maybe_skip(x, lambda xt: step_fn(xt, t_in, cond))
            x = x + v * dt

        # Stash backbone call count for tests / profiling when CFG was used.
        if wrapped is not None:
            self.last_backbone_calls = wrapped.call_count
        else:
            self.last_backbone_calls = self.num_steps
        return x


__all__ = ["FlowMatching", "CFGWrapper"]
