# ProPainter MLX port — spec

## Goal

Port the **ProPainter video-inpainting pipeline** from PyTorch to MLX so we
can do video inpainting on Apple Silicon GPU, no torch / no CUDA. Designed
to feed our `meadow_sb` pipeline (remove person/object → fill background →
3DGS the clean scene).

Inference-only. We do not port training code.

## Source

- Upstream code: `/Users/akaihuangm1/Desktop/github/ProPainter/`
- Weights: `/Users/akaihuangm1/Desktop/github/propainter-mlx/weights/propainter-pt/`
  - `ProPainter.pth` (160 MB, 39.4 M params, 216 tensors) — main inpainter
  - `recurrent_flow_completion.pth` (19 MB) — flow completion
  - `raft-things.pth` (20 MB) — RAFT optical flow
  - `i3d_rgb_imagenet.pt` (49 MB) — for perceptual loss (training-only, skip)
- Paper: https://arxiv.org/abs/2309.03897 "ProPainter: Improving Propagation and Transformer for Video Inpainting"

## Architecture overview

Three sub-models, run in sequence:

### 1. RAFT (optical flow, 20 MB)

Standard pre-trained RAFT. Given two RGB frames, output 2-channel flow.

```
inputs:  (B, 3, H, W) prev + (B, 3, H, W) curr
output:  (B, 2, H, W) flow vectors (px)
```

Architecture: feature encoder (residual conv) + context encoder + GRU update.
Many MLX RAFT ports exist; we should reference one if it speeds up the port.

### 2. RecurrentFlowCompletion (RFC, 19 MB)

Fills in flow vectors at masked regions using a recurrent network over time.

```
inputs:  flow + mask sequences
output:  completed flow sequences (mask region filled)
```

### 3. ProPainter main (160 MB, 39 M params)

Top-level structure from PT state-dict prefixes:
- `transformers.*` (152 tensors)   ← bulk of params
- `feat_prop_module.*` (32)        ← flow-guided feature propagation
- `encoder.*` (18)                 ← downsample + tokenise
- `decoder.*` (8)                  ← upsample + reconstruct
- `sc.*` (4), `ss.*` (2)          ← skip-connection adjustment

Conceptual flow:

```
input frames (B, T, 3, H, W) + masks (B, T, 1, H, W)

encoder → tokens (B, T, h*w, C)

flow_completion (RFC + RAFT) → flow tokens (B, T-1, 2, h, w)

feat_prop_module → propagate tokens along flow trajectories
   - forward propagation
   - backward propagation
   - bidirectional fusion

transformers (~7 blocks) → temporal + spatial attention
   - inter-frame attention over propagated tokens
   - intra-frame self-attention
   - feed-forward

decoder → upsample back to (B, T, 3, H, W) RGB

output = mask * inpainted + (1 - mask) * input
```

## MLX porting notes

- **3D conv**: ProPainter uses Conv3d in encoder for temporal context. MLX
  has Conv3d in 0.31+; verify it's stable.
- **Deformable conv / flow warping**: `grid_sample` (PT) → need MLX equivalent;
  MLX has `mx.gather` but not bilinear grid sample. Port via bilinear
  interpolate from indices.
- **Attention masking**: long-range temporal attention across all frames.
  Memory scales with T². Plan: chunk into windows of T=8 or 16 if needed.
- **RAFT**: simpler than ProPainter main; port first as a warm-up.

## Weight loading

3 separate .pth files. Each loaded via:
1. `torch.load(weights_only=True)` (RAFT + RFC are clean state dicts)
2. ProPainter.pth — check if pl-bundled; same stub trick as LaMa if so.
3. Walk PT keys, strip prefixes, transpose Conv weights, save flat npz.

## File layout (target)

```
propainter-mlx/
├── SPEC.md                          (this file)
├── README.md                        (user-facing, write last)
├── pyproject.toml
├── propainter_mlx/
│   ├── __init__.py
│   ├── raft.py                      RAFT MLX port (used by both RFC and main)
│   ├── flow_completion.py           RecurrentFlowCompletion
│   ├── feat_prop.py                 Feature propagation along flows
│   ├── transformers.py              Temporal + spatial attention blocks
│   ├── encoder.py                   Encoder (Conv3d → tokens)
│   ├── decoder.py                   Decoder (tokens → frames)
│   ├── propainter.py                Top-level assembler
│   └── weights.py                   PT -> MLX npz loader
├── scripts/
│   ├── dump_pt_activations.py       per-block PT forward dump
│   ├── convert_weights.py           pth -> npz
│   └── inpaint_video_cli.py         CLI: video + masks → inpainted video
├── tests/
│   ├── test_raft.py
│   ├── test_flow_completion.py
│   ├── test_feat_prop.py
│   ├── test_transformers.py
│   └── test_e2e.py                  end-to-end vs PT reference
└── weights/
    └── propainter-pt/               symlink to PT weights
```

## Quality gate

- RAFT: end-point error vs PT < 0.1 px (typical EPE is ~1 px on KITTI)
- RFC: completed flow MAE < 1 px in mask region
- Main inpainter per-block: max |out| diff < 5e-3 fp32
- End-to-end: PSNR vs PT inpainted video > 35 dB

## Estimated effort

- RAFT port: 2-3 days
- RFC port: 1-2 days
- Encoder + decoder: 1-2 days
- Feat-prop module: 2 days (grid_sample is the trickiest)
- Transformers: 2 days
- End-to-end assembly + test: 1 day
- CLI + README: 1 day
- Total: ~12 days (≈ 2 weeks)

## Integration with meadow

```
video.mp4 → SAM3-mlx → per-frame masks
         → ProPainter-mlx → clean (mask-removed) video
         → meadow_sb (existing) → 3DGS .ply
```

This pipeline gives the "remove person/object, recover pure spatial 3DGS"
flow the user wants.
