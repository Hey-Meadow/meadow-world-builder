# SAM 3D Objects → MLX Port: Recon Plan

## Goal
Map the 20k LoC architecture before any porting. Three independent recon agents.

## Why recon first
SAM 3D Body was a clean port (4k LoC, ViT + decoder, regression head).
SAM 3D Objects is **5× larger** with:
- DiT (Diffusion Transformer)
- Flow matching sampler (multi-step generative)
- pytorch3d / kaolin (CUDA-only mesh + render libs)
- Sparse representations (octree, gaussian, flexicubes mesh, radiance field)

We need to know the inference critical path before porting. Don't waste effort on training-only / dataset / loss code.

## Three recon agents

### Agent OBJ-1: Backbone & DiT architecture
- Map `sam3d_objects/model/backbone/dit/` (regular DiT)
- Map `sam3d_objects/model/backbone/tdfy_dit/` (the bigger 3D-aware DiT)
- Identify transformer blocks, attention variants, conditioning mechanisms
- Output: `RECON_DIT.md` listing module hierarchy + LoC + reusable-from-Body parts

### Agent OBJ-2: 3D representations & rendering
- Map sparse conv, sparse attention, octree, gaussian, flexicubes mesh, radiance field
- Identify pytorch3d / kaolin usage points
- Output: `RECON_3D.md` listing every CUDA-tied op with feasibility notes

### Agent OBJ-3: Inference path & sampler
- Trace `demo.py` end-to-end: image+mask → 3D output
- Map `generator/flow_matching/`, `generator/shortcut/`, `classifier_free_guidance.py`
- Identify what runs ONCE vs per-sampling-step
- Output: `RECON_INFER.md` with critical path timing budget (% per stage)

## After recon (later phase)
Based on RECON_*.md, we'll dispatch:
- Port-DiT agent (uses already-built MLX ViT patterns)
- Port-sampler agent (flow matching)
- Pytorch3d/kaolin replacement agents (mesh extract, rendering)

## Constraints for recon agents
- **READ ONLY**. Do not write any model code yet.
- Look at file sizes, imports, class hierarchies, NOT every line.
- Honest assessment: if a module is too tangled to port, say so.
- Each report ≤ 500 words.
