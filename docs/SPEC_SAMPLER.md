# SPEC_SAMPLER.md — Agent OBJ-SAMPLER (Flow matching ODE + CFG)

## Goal
Port the flow matching Euler ODE sampler + Classifier-Free Guidance wrapper from `sam3d_objects/model/backbone/generator/` to MLX.

## Inputs
- **PT source**: `sam3d_objects/model/backbone/generator/{base.py, flow_matching/, classifier_free_guidance.py}`
- **Plan**: `mlx_port/docs/PORT_PLAN.md`
- **Recon**: `mlx_port/docs/RECON_INFER.md` (read first — explains sampling loop)
- **Reference**: SAM 3D Body code patterns

## What this module does (per RECON_INFER.md)
1. Initialize random latent at `t=1`
2. Loop N steps (default 25):
   - Run DiT forward at time `t` with conditioning → predicted velocity
   - If CFG enabled (default): also run with empty conditioning, blend
   - Euler step: `x_{t-dt} = x_t - v * dt`
3. Return final `x_0`

## Required deliverables

### 1. `mlx_port/models/sampler_mlx.py`
```python
import mlx.core as mx
import mlx.nn as nn
from typing import Callable, Optional

class FlowMatching:
    def __init__(self, num_steps: int = 25, sigma_max: float = 1.0,
                 cfg_strength: float = 7.0, schedule: str = "linear"): ...

    def sample(
        self,
        backbone_fn: Callable[[mx.array, mx.array, mx.array], mx.array],
        cond: mx.array,
        uncond: Optional[mx.array] = None,
        shape: tuple = None,           # latent shape (B, N, C)
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        """
        backbone_fn signature: (latent, t, cond) -> velocity
        Returns: final x_0 latent (B, N, C)
        """
        ...

class CFGWrapper:
    """Wrap a backbone forward such that CFG is applied (cond+uncond blend)."""
    def __init__(self, backbone, cfg_strength: float, uncond_token: mx.array): ...
    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array: ...
```

### 2. `mlx_port/tests/test_sampler.py`
- Mock backbone (returns negated input — trivial ODE → identity).
- Sampler run with CFG strength 0 (no CFG) — verify output shape.
- Sampler with CFG strength 7 (calls backbone twice per step) — verify call count.

## Notes
- **CFG cost**: each step calls backbone TWICE if cfg_strength > 0. This is the main inference cost multiplier.
- **Schedule**: probably linear time schedule — check PT source for `schedule_t = linspace(1, 0, N+1)`.
- **Pointmap CFG variant** (3-way blend with pointmap-only branch): only used if `ss_cfg_strength_pm > 0`. Default is 0, so START with simple 2-way CFG, can add pointmap variant later.
- **No model weights needed for sampler itself** — it's just an ODE loop. The backbone is plugged in via `backbone_fn`.

## Constraints
- Pure MLX, no torch dependency.
- Use `mx.linspace` for time schedule, `mx.array` slicing.
- DO NOT modify `sam3d_objects/` or `sam-3d-body/` (read-only).
- Working dir: `/Users/akaihuangm1/Desktop/github/sam-3d-objects/`
- Python: `/Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python`

## Definition of done
1. `sampler_mlx.py` imports cleanly.
2. Test passes (mock backbone path).
3. Brief report (≤ 200 words) including:
   - Time schedule used (linear / quadratic / etc., from PT source)
   - CFG strength default (7 for shape, 5 for texture per RECON_INFER.md)
   - Any non-trivial details from PT source (e.g., specific noise prediction parametrization)
