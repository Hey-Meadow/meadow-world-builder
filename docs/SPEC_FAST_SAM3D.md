# Fast-SAM3D Port Spec (Meadow - World Builder Adaptation)

Source paper: **Fast-SAM3D: 3Dfy Anything in Images but Faster**, arXiv [2602.05293](https://arxiv.org/abs/2602.05293).

Training-free. Author-reported 2.67× end-to-end on Toys4K (single-object).
**Mechanism ③ targets the mesh decoder, which Meadow - World Builder does NOT use** — we
ship only the Gaussian-splat path. So our realistic gain is from ① and ②
only.

---

## Applicability Matrix (Meadow - World Builder pipeline)

| Mechanism | Paper target | Meadow - World Builder stage | Applies? | Expected gain (M1 Max) |
|---|---|---|---|---|
| ① Modality-aware step caching | SS Generator (25-step CFG) | SS DiT (already 4-step shortcut) | **partial** — see §1 below | ~5–10% on SS (~0.5–1s) |
| ② Joint spatiotemporal token carving + curvature caching | SLAT Generator (25-step CFG) | SLAT DiT (still 25-step CFG-5, ~80 s) | **YES — primary target** | ~34% of SLAT (~25–27 s) |
| ③ Spectral-aware token aggregation | Mesh decoder | (none — we use GS decoder) | **NO** | 0 |

**Realistic end-to-end after ①+②**: 100 s → ~72 s (≈1.4×).
The paper's 2.67× number includes ③ on the mesh path, which we lose by
not shipping mesh.

---

## ① Modality-Aware Step Caching (SS Generator)

### Idea
Decouple two token streams in the SS Generator: **shape** tokens
(geometric layout, smooth evolution) and **layout** tokens (object
pose / placement, high-frequency volatility). Cache each with a
different strategy.

### Equations

**Finite-difference anchor (shape tokens)** — local gradient from two
full evaluations $k$ steps apart:

$$\nabla\mathbf{v}^{\text{shape}}_{t} = \frac{\mathbf{v}^{\text{shape}}_{t} - \mathbf{v}^{\text{shape}}_{t+k}}{k}$$

For skipped step $t-i$ ($1 \leq i < k$), extrapolate via 1st-order
Taylor:

$$\hat{\mathbf{v}}^{\text{shape}}_{t-i} = \mathbf{v}^{\text{shape}}_{t} + (-i)\nabla\mathbf{v}^{\text{shape}}_{t}$$

**Momentum-anchored smoothing (layout tokens)** — blend linear
extrapolation with last anchor:

$$\hat{\mathbf{v}}^{\text{layout}}_{t-i} = \beta \cdot \mathbf{v}^{\text{layout}}_{\text{lin}}(t-i) + (1-\beta) \cdot \mathbf{v}^{\text{layout}}_{\text{anchor}}$$

where $\mathbf{v}^{\text{layout}}_{\text{anchor}}$ is the most-recent
full backbone evaluation.

### Hyperparameters (paper defaults)

| Param | Default | Sensitivity |
|---|---|---|
| $k$ (cache stride) | **3** | $k \geq 4$ → catastrophic failure (3D-IoU 0.375 → 0.24) |
| $\beta$ (momentum factor) | **0.5** | optimum; $\beta=1.0$ slightly worse |
| Warmup | **2 steps** | required for stability |

### Meadow - World Builder-specific risk
Our SS already runs **4-step shortcut** distillation (`--use-shortcut`).
The paper assumes a 25-step baseline; with only 4 steps there's almost
no headroom to skip. Two paths:

- **Path A (recommended)**: only apply ① when `--no-shortcut` is set
  (the 25-step path), preserve shortcut as the default.
- **Path B**: try $k=2$ on the 4-step shortcut — at best saves 1 of 4
  steps. Probably below noise floor.

### How to identify shape vs layout tokens?
Paper doesn't make this trivial. In SAM 3D Objects, the SS Generator
output is a structured-token grid. Treat the **occupancy logit token**
as "layout" and the **feature tokens** as "shape" — that matches the
paper's framing (occupancy = drift-prone placement; features = smooth
geometric continuation). Validate empirically with the ablation in
§Validation Plan below.

### Pseudocode

```python
class ModalityAwareCache:
    def __init__(self, k: int = 3, beta: float = 0.5, warmup: int = 2):
        self.k, self.beta, self.warmup = k, beta, warmup
        self.shape_anchor = None
        self.shape_grad = None
        self.layout_anchor = None
        self.layout_grad = None
        self.step = 0

    def maybe_cache(self, t: int, run_full_fn):
        if self.step < self.warmup or self.step % self.k == 0:
            v_shape, v_layout = run_full_fn(t)
            if self.shape_anchor is not None:
                k = self.step - (self.step - self.k)
                self.shape_grad  = (v_shape  - self.shape_anchor)  / k
                self.layout_grad = (v_layout - self.layout_anchor) / k
            self.shape_anchor, self.layout_anchor = v_shape, v_layout
            self.step += 1
            return v_shape, v_layout
        # Skip — extrapolate
        i = self.step - (self.step // self.k) * self.k   # offset into cache window
        v_shape_hat  = self.shape_anchor + (-i) * self.shape_grad
        v_layout_lin = self.layout_anchor + (-i) * self.layout_grad
        v_layout_hat = self.beta * v_layout_lin + (1 - self.beta) * self.layout_anchor
        self.step += 1
        return v_shape_hat, v_layout_hat
```

---

## ② Joint Spatiotemporal Token Carving + Curvature-Aware Caching (SLAT)

**This is the highest-leverage mechanism for Meadow - World Builder** — SLAT is 80 %
of our wall time and runs the full 25-step CFG-5 schedule (no shortcut
yet).

### Idea
Two simultaneous compute reductions on the SLAT DiT denoising loop:

1. **Spatial**: at each step, only refine the top-$K\%$ tokens that
   matter, score-ranked by a 3-term saliency.
2. **Temporal**: when the velocity-field curvature is low (quasi-linear
   regime), reuse the previous step's update as a tangent — skip the
   forward pass entirely until accumulated error exceeds threshold.

### Spatial saliency

Per-token score combines magnitude, abrupt change, and high-frequency
spatial energy:

$$\mathcal{M}_i(t) = \|\mathbf{v}_{t,i}\|_2$$
$$\mathcal{A}_i(t) = \|\mathbf{v}_{t,i} - \mathbf{v}_{t+1,i}\|_2$$
$$\mathcal{S}_{\text{freq}}(i) = \frac{\sum_{\omega \in \Omega_{\text{high}}}\|\mathcal{F}(\mathbf{v}_i)[\omega]\|^2}{\sum_\omega \|\mathcal{F}(\mathbf{v}_i)[\omega]\|^2}$$

$$\mathcal{J}_i(t) = \tfrac{1}{2}\!\left(\mathcal{M}_i(t) + \gamma\mathcal{A}_i(t)\right) + \tfrac{1}{2}\mathcal{S}_{\text{freq}}(i)$$

Keep the top-$K\%$ tokens by $\mathcal{J}_i$ in the active set. Dropped
tokens hold their last-known velocity (no forward through DiT block).

### Temporal curvature

Trajectory curvature:

$$\kappa_t = \frac{\|\mathbf{v}_t - \mathbf{v}_{t-1}\|_2}{\|\mathbf{x}_t - \mathbf{x}_{t-1}\|_2}$$

Tangent reuse:

$$\Delta_i := \mathbf{v}_i - \mathbf{x}_i, \quad \hat{\mathbf{v}}_t = \mathbf{x}_t + \Delta_i$$

Accumulated error since last anchor:

$$E_t = \sum_{n=i+1}^{t} \frac{\|\mathbf{v}_n - \mathbf{v}_{n-1}\|_2}{\|\mathbf{v}_{n-1}\|_2}$$

If $E_t \geq \mathcal{E}$ — anchor refresh (full forward). Otherwise —
reuse tangent.

### Hyperparameters

| Param | Default | Notes |
|---|---|---|
| $K$ (carving %) | **10 %** | of active tokens retained per step |
| $\gamma$ (carving factor) | **0.7** | weight on abrupt-change term |
| $\mathcal{E}$ (switching threshold) | **1.5** | error bound triggering anchor refresh |
| Warmup | **2 steps** | both spatial and temporal disabled |

### Meadow - World Builder-specific risk
- SLAT has only **16 000 voxel tokens max**, not 100k+. With $K=10\%$
  that's only 1 600 active tokens per step. Risk of under-fitting
  geometry on complex objects (plush). **Ablate $K \in \{10, 20, 30, 50\}$%.**
- FFT cost per step on 16 000 sparse tokens needs benchmarking — could
  itself eat the savings.
- Curvature-aware caching combined with shortcut model untested in
  paper. We only have it on SS, not SLAT — so safe for SLAT.

### Pseudocode

```python
class SpatiotemporalCarving:
    def __init__(self, K_pct=0.10, gamma=0.7, E_thresh=1.5, warmup=2):
        self.K, self.gamma, self.E_thresh, self.warmup = K_pct, gamma, E_thresh, warmup
        self.prev_v = None
        self.last_anchor_v = None
        self.last_anchor_x = None
        self.E_acc = 0.0
        self.step = 0

    def step_slat(self, x_t, run_dit_block_fn):
        # ---- Temporal: skip whole forward if error budget not exhausted ----
        if self.step >= self.warmup and self.E_acc < self.E_thresh and self.last_anchor_v is not None:
            delta = self.last_anchor_v - self.last_anchor_x
            v_t = x_t + delta
            # update error budget
            self.E_acc += mlx.linalg.norm(v_t - self.prev_v) / (mlx.linalg.norm(self.prev_v) + 1e-8)
            self.prev_v = v_t
            self.step += 1
            return v_t
        # ---- Spatial: rank tokens, only refine top-K% ----
        mag  = mlx.linalg.norm(self.prev_v, axis=-1) if self.prev_v is not None else None
        abrp = mlx.linalg.norm(self.prev_v - x_t, axis=-1) if self.prev_v is not None else None
        freq = high_freq_energy_ratio(self.prev_v)  # FFT per-token; warmup-only None
        if mag is None:
            v_t = run_dit_block_fn(x_t, mask=None)        # warmup full pass
        else:
            J = 0.5 * (mag + self.gamma * abrp) + 0.5 * freq
            topk = int(self.K * J.shape[0])
            active_idx = mlx.argsort(J)[-topk:]
            v_t = self.prev_v.copy()
            v_t[active_idx] = run_dit_block_fn(x_t[active_idx], mask=active_idx)
        # anchor refresh
        self.last_anchor_v = v_t
        self.last_anchor_x = x_t
        self.E_acc = 0.0
        self.prev_v = v_t
        self.step += 1
        return v_t
```

---

## ③ Spectral-Aware Token Aggregation — DEFER

Targets mesh decoder. Meadow - World Builder ships Gaussian-splat path only (no mesh
decoder in `gs_4` decode). Possible future direction: adapt HFER to
choose `gs_4 / gs_8 / gs_16` per object based on input complexity, but
this is research, not a port. **Not in scope for this sprint.**

---

## Integration Points (where each mechanism plugs in)

| Mechanism | File to modify | Function to wrap |
|---|---|---|
| ① Modality-aware caching | `meadow_wb/models/sampler_mlx.py` | SS sampler loop |
| ② Spatiotemporal carving | `meadow_wb/models/sampler_mlx.py` | SLAT sampler loop |
| ② curvature caching | `meadow_wb/models/dit_mlx.py` | wrap `slat_dit.forward` call |
| ② FFT saliency | new `meadow_wb/utils/saliency.py` | per-token HFER |

CLI flags to add to `meadow_wb/infer.py`:

```
--fast-sam3d                       enable ①+②
--ss-cache-k INT                   default 3
--ss-cache-beta FLOAT              default 0.5
--slat-carve-pct FLOAT             default 0.10
--slat-carve-gamma FLOAT           default 0.7
--slat-curvature-eps FLOAT         default 1.5
--no-slat-carving                  disable carving (keep only curvature caching)
--no-ss-modality-cache             disable ① (keep only ②)
```

---

## Validation Plan (must pass before merging)

Run on `assets/demos/{chair,table,plush}.ply` rebuilt from their
respective input images (under `mlx_port/debug/input/<obj>/`).

| Gate | Metric | Tolerance |
|---|---|---|
| **Speed** | end-to-end wall time | ≥ 1.3× speedup on at least 2 of 3 objects |
| **Geometry** | bbox(x,y,z) Δ vs current MLX output | within ±5% |
| **Quality** | Gaussian count Δ | within ±10% |
| **Quality** | opacity mean / median Δ | within ±0.03 |
| **Quality** | `|q|` (quaternion norm) | exactly 1.0000 (no regression) |
| **Visual** | side-by-side render (4 azimuths) | no obvious artifacts |

Failure on any gate → rollback the responsible mechanism, ablate, fix.

---

## Implementation Order

1. **Cache scaffolding** — `ModalityAwareCache` + `SpatiotemporalCarving`
   classes in `meadow_wb/utils/cache.py`. Wire CLI flags. *(~1 day)*
2. **Mechanism ②a (curvature caching only)** — temporal only, no
   spatial carving. Easiest validation. Target: 1.2× on SLAT alone.
   *(~2 days)*
3. **Mechanism ②b (spatial carving)** — layer FFT saliency + top-K
   gating on top of ②a. Validate quality gate carefully — this is the
   highest-risk mechanism. *(~2 days)*
4. **Mechanism ①** — only after ② is green. Lower payoff for us
   (shortcut already applied), but cheap to add. *(~1 day)*
5. **Final benchmark** — full ablation table, write up. *(~1 day)*

**Total: ~7 working days** for ①+② combined, matching the table's "1 week" estimate.

---

## Open Questions

1. **Shape vs layout token split in SAM 3D Objects' SS DiT** — paper
   doesn't say explicitly which channels are which. First debug step:
   dump per-channel mean update magnitude over 25 steps from a PT run,
   look for the smooth/volatile dichotomy.
2. **FFT cost on M1 Max for 16 000 × `d_token` tensors** — needs
   microbenchmark; if it's > 5 % of SLAT step time, we lose the savings.
3. **Interaction with `--dtype mixed`** — paper runs fp32. Cache
   feature anchors in bf16 should be fine (small error per step,
   accumulated across only ~k steps) but needs check.
