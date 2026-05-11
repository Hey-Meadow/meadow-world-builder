# YoNoSplat MLX Port — Spec Research

Branch: `feature/space-port-research`
Upstream: [cvg/yonosplat](https://github.com/cvg/yonosplat) (MIT License)
Project page: <https://botaoye.github.io/yonosplat/>
Paper: [arXiv 2511.07321](https://arxiv.org/abs/2511.07321)
Sibling product: [meadow-world-builder](https://github.com/Hey-Meadow/meadow-world-builder) handles single-object → 3DGS; **YoNoSplat handles multi-image → scene-scale 3DGS + camera poses.** Together = full Meadow 3D capture suite.

## What YoNoSplat does (verified by reading source)

| Aspect | Value |
|---|---|
| Input | **N RGB images** (2–32, unposed and uncalibrated) |
| Output | scene-level 3D Gaussian Splat + **6-DoF camera poses per input view** |
| Foundation model | [Pi3](https://huggingface.co/yyfz233/Pi3) (CroCo-multiview ViT-L, 24 enc + 12 dec layers) |
| Resolution | 224 × 224 (released checkpoints) |
| Datasets trained on | RealEstate10K (`re10k.ckpt`), DL3DV (`dl3dv.ckpt`) |
| License | **MIT** — commercial use OK with attribution |

YoNoSplat is the multi-view, scene-scale, pose-free counterpart to single-image SAM 3D Objects. It eats N photos and emits a unified GS plus camera poses; we already shipped the single-image path.

## Module map — what we actually have to port

LOC counts measured on the cloned repo. Critical inference path only (excludes training, loss, visualisation, dataset loaders, hydra/lightning glue).

| Module | LOC | What it does | MLX port effort |
|---|---:|---|---|
| `src/model/encoder/backbone/dinov2/` | ~1500 | DINOv2 ViT (xformers attention, SwiGLU FFN) | **Medium** — we already did DINOv3-MLX in sam-3d-objects; same playbook |
| `src/model/encoder/backbone/croco/` | ~700 | CroCo cross-view completion blocks (RoPE positional, dual-branch attention) | **High** — new architecture, no prior MLX port |
| `src/model/encoder/backbone/backbone_croco.py` | 280 | CroCo wrapper + Pi3 weight loading | Low — config + I/O |
| `src/model/encoder/layers/` (attention/block/heads) | ~1100 | Multi-view transformer, camera head, transformer head, upscale head | Medium — std transformer ops |
| `src/model/encoder/encoder_yonosplat.py` | 336 | Top-level encoder: backbone + heads → Gaussians + poses | Medium |
| `src/model/decoder/decoder_splatting_gsplat.py` | 100 | Gaussian rasterizer interface | Wrapper — see "Rasterizer" section below |
| `src/model/decoder/cuda_splatting.py` | 230 | CUDA-only `diff_gaussian_rasterization` path | **High** — CUDA only, we replace not port |
| `src/model/decoder/prune.py` | 68 | Opacity-threshold prune of low-α Gaussians | Trivial |
| `src/geometry/` | ~1000 | Projection, intrinsic embedding, FoV, camera maths | Medium — pure tensor ops |
| **Total inference path** | **~7800 LOC** | | |

For context: `sam-3d-objects` MLX port was ~5 000 LOC and took ~3 weeks. YoNoSplat is ~1.5× bigger with one entirely new architecture family (CroCo) and a hard rasterizer dependency.

## Hard dependency analysis

```
PyTorch 2.1.2 + CUDA 11.8 (upstream pinned)
├── xformers                                  # → mx.fast.scaled_dot_product_attention (we know this drill)
├── diff-gaussian-rasterization-w-pose        # CUDA forward + backward — INFERENCE only path: rewrite forward in Metal
├── gsplat                                    # alternative rasterizer; pure-Python rasterize_to_pixels has CPU fallback in v1.4+ (verify before counting on it)
├── timm                                      # ViT helpers — replace with manual MLX modules
├── e3nn                                      # equivariant SO(3) layers for pose — small surface area, port directly
├── lightning + hydra + omegaconf             # training/config infra — skip for inference port
├── beartype + jaxtyping                      # runtime type checks — skip
└── wandb                                     # training logger — skip
```

**Two blocking dependencies for an Apple-Silicon-only build:**

1. **`diff_gaussian_rasterization`** — CUDA-only kernel. *Inference does not need backward.* The forward operation is well-known:
   - Project each Gaussian's 3D centre into screen space (camera intrinsics)
   - Compute 2D screen-space covariance from 3D scale + rotation + camera Jacobian
   - Sort splats by view-space depth
   - Tile-based alpha composite with `(1 − α_dst) · α_src` blending
   We rewrite this as a Metal compute kernel. Reference: the original 3DGS paper's CUDA code (~1500 LOC; forward only is ~500). Estimate **2–3 weeks** for a working forward-only Metal version.

2. **xformers** — already a solved problem from `sam-3d-objects`. Apply the same swap: `xformers.ops.memory_efficient_attention` → `mlx.fast.scaled_dot_product_attention`. Estimate **~3 days** across all DINOv2 + CroCo files.

## Phased port plan

### Tier 1 — MVP inference (5–6 weeks)

Goal: produce a runnable `infer.py` that takes 2 unposed images and outputs a `.ply` + camera poses, comparable to upstream's `re10k.ckpt` output quality.

| Phase | What | Effort |
|---|---|---|
| 0. RunPod reference dump | A100 reproduce upstream, save per-stage activations + final `.ply` for numerical validation. Mirror the `dump_pt_reference.py` flow we used on sam-3d-objects. | 3 days |
| 1. DINOv2 backbone → MLX | Reuse 80% of `sam-3d-objects` ViT port; swap depth + dim configs. | 4 days |
| 2. CroCo cross-view blocks → MLX | New module: RoPE positional, cross-attention between views. | 7 days |
| 3. YoNoSplat encoder head → MLX | Camera head, upscale head, Gaussian regression head. | 4 days |
| 4. Pure-MLX/gsplat-CPU rasterizer hookup | Use `gsplat` CPU path if available; otherwise CPU NumPy fallback. Slow but unblocks end-to-end. | 3 days |
| 5. Weight conversion script | PT → MLX npz, validate per-layer activation deltas vs RunPod dump. | 4 days |
| 6. End-to-end + validation | Run on RE10K val set; quality + per-stage timing report. | 4 days |

### Tier 2 — production speed (3–4 weeks on top of Tier 1)

| Phase | What | Effort |
|---|---|---|
| 7. Metal forward-only Gaussian rasterizer | Replace the gsplat-CPU fallback. ~500 LOC of Metal compute shader. | 14 days |
| 8. bf16 mixed precision | DiT blocks in bf16, accumulators in fp32. Reuse the `--dtype mixed` plumbing from `sam-3d-objects`. | 3 days |
| 9. Curvature cache analogue (if applicable) | YoNoSplat is feed-forward (no diffusion sampling loop), so curvature cache may not apply. Investigate cross-view attention reuse instead. | TBD |
| 10. Public HF weights upload | Same `akaiii/meadow-world-builder-weights` pattern. | 1 day |

### Tier 3 — full feature parity (optional)

| Phase | What |
|---|---|
| 11. Backward pass for fine-tuning on user data | Lets users adapt to their own scene types |
| 12. Live WebGL viewer for scenes | Extend `feature/iridescent-shader` from object splats to scene splats |
| 13. Dynamic scenes / video input | Treat video as N-view sequence with temporal regularisation |

## First-step actions (this branch)

1. **`docs/PORT_PLAN_YONOSPLAT.md`** ← this file
2. **Mirror the dependency graph in this repo** — add `docs/YONOSPLAT_MODULE_MAP.md` with a per-file annotation of upstream → planned MLX target (similar to `mlx_port/docs/PORT_PLAN.md` in sam-3d-objects)
3. **Stand up a clean RunPod reference run** — `pip install -r requirements.txt` on A100, run a known image pair through `re10k.ckpt`, save inputs + outputs + intermediate activations. ~$5 of compute.
4. **License audit on Pi3** — confirm `yyfz233/Pi3` HF page allows commercial redistribution of derivative MLX weights.
5. **Open the port repo** — fork `cvg/yonosplat` to `Hey-Meadow/meadow-space-builder` or create `meadow-space-builder` afresh that imports the upstream as a git submodule.

## Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Metal rasterizer underperforms CUDA by 10×+ | Medium | High | Optimise tile sizes empirically; fall back to CPU NumPy for correctness then revisit |
| 2 | CroCo cross-attention has Apple-Silicon-specific dtype quirks (RoPE precision) | Medium | Medium | Validate per-block with PT reference dump (the strategy that worked on sam-3d-objects) |
| 3 | Pi3 weights are MIT but trained on data with redistribution restrictions | Low | High | Cite Pi3 explicitly + don't redistribute weights (mirror our SAM License approach) |
| 4 | `gsplat` CPU path doesn't exist or is broken | High | Low | Just write the Metal rasterizer (which we need anyway for Tier 2) |
| 5 | Multi-view scene-scale GS quality regresses on Apple-Silicon-FP16 vs CUDA-FP32 | Medium | Medium | bf16 by default, fp32 fallback flag; validate against RE10K test split |

## Why this is worth doing

- **Strategic complement to meadow-world-builder.** Sister product completes the "Meadow 3D capture" portfolio: objects (done) + scenes (this).
- **No commercial-friendly Apple-Silicon multi-view 3DGS exists today.** Both YoNoSplat (MIT) and our port (Apache 2.0 derivative + MIT upstream) are commercial-usable, on-device, no cloud.
- **Multi-view fusion is the negative result we got from naïve ICP** (`feature/multi-view-merge` branch). Doing it properly via a trained model is the right path.
- **Same dev tooling reused.** Weight-conversion script, RunPod reference-dump methodology, sub-agent port playbook — all carried over from `sam-3d-objects` port. Probably 30 % less calendar time than starting from scratch.

## What we do NOT need to port

- Training pipeline (lightning, hydra, datasets) — pull releases instead
- Visualisation (epipolar visualizer, drawing utilities) — Meadow's own WebGL viewer covers this
- Dataset shims / view samplers — runtime path needs none
- Evaluation harnesses — port reference behaviour, not benchmarking infra
- DPT / depth / pose-completion heads not active for the released `re10k.ckpt` checkpoint

Total skip ≈ 60 % of upstream codebase by LOC.
