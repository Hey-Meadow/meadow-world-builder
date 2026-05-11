# RunPod Reference-Run Plan — YoNoSplat MLX port validation

**Status:** not executed yet. This is the plan the user will run once the
MLX port is integration-ready (post Agent A-G landing, after `e2e_test.py`
shows non-trivial backbone diffs to compare against). Owner: Agent H.

## 1. Purpose — what only RunPod can give us

Local M1 Max can do everything except the two things below. Those two
artifacts are the *only* reasons we spin up a GPU pod for the YoNoSplat
port:

| Need | Why local fails | RunPod gives |
|---|---|---|
| Ground-truth **rendered images** for the canonical 2-view input | `diff-gaussian-rasterization-w-pose` is a CUDA-only extension; the local stub in `/tmp/yonosplat_inspect` returns zeros | Real splatted RGB at the pose YoNoSplat predicts → PSNR / LPIPS reference |
| **Clean speed baseline** (forward latency, peak VRAM) | M1 wall-clock conflates MLX overhead, MPS quirks, and Python-side glue | A100 reference number (`xx ms / 2-view pair`) to anchor the "MLX is k× of A100" claim |

Everything else — per-block activation comparison, weight conversion sanity
checks, MLX shape gating — already lives in
`research/yonosplat_bootstrap/dumps/per_block/` and is reproduced by
`meadow_sb/scripts/e2e_test.py`.

## 2. Pod selection

- GPU: **1 × A100 80 GB SXM** (or A100 PCIe 40 GB — re10k fits in 40 GB).
- Image: `runpod/pytorch:2.1.2-py3.10-cuda11.8.0-devel-ubuntu22.04`
  - Pinned because `diff-gaussian-rasterization-w-pose` builds reliably
    on CUDA 11.8 + PyTorch 2.1.x; CUDA 12 wheels are flakier.
- Disk: >= 40 GB (`re10k.ckpt` is 3.86 GB, upstream + deps ~6 GB, headroom).
- Networking: SSH only; no extra ports.

## 3. Exact commands

### 3.1 One-time setup (~5 min)

```bash
# On the pod, fresh shell ----------------------------------------------------
cd /workspace
apt-get update && apt-get install -y git build-essential
git clone --depth=1 https://github.com/Lakonik/YoNoSplat.git yonosplat
cd yonosplat

# Pin upstream commit — replace HASH with the one we resolved locally
# (the same commit that produced research/yonosplat_bootstrap/dumps/).
git checkout HASH

# Python deps (matches /tmp/yonosplat_inspect that produced our local dumps)
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir \
    "git+https://github.com/ashawkey/diff-gaussian-rasterization-w-pose@main"

# Sanity: import the CUDA extension once so wheel build errors surface now.
python -c "import diff_gaussian_rasterization as dgr; print(dgr.__file__)"
```

### 3.2 Stage the reference inputs

```bash
# Pull canonical inputs from local M1 (run from local):
DEST=pod:/workspace/ref/
ssh pod 'mkdir -p /workspace/ref'
scp research/yonosplat_bootstrap/weights/yonosplat/re10k_224x224_ctx2to32.ckpt "$DEST"
scp research/yonosplat_bootstrap/dumps/test_input.npz                          "$DEST"
```

> Same `test_input.npz` we used locally. Critical — RunPod renders must
> share inputs with our local dumps, otherwise pixel-level comparison is
> meaningless.

### 3.3 Reference run (~30 s wall, 1 forward pass)

`runpod_ref.py` — write to `research/yonosplat_bootstrap/scripts/` later,
then `scp` to `/workspace/`. Sketch below; the precise encoder-construction
helper mirrors upstream `demo.py`:

```python
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/workspace/yonosplat")

CKPT = "/workspace/ref/re10k_224x224_ctx2to32.ckpt"
INP  = "/workspace/ref/test_input.npz"
OUT  = "/workspace/ref/out"; os.makedirs(OUT, exist_ok=True)

# 1) Build the full EncoderYoNoSplat via the Hydra config the demo uses.
from src.model.encoder.encoder_yonosplat import EncoderYoNoSplat
encoder = build_encoder_from_config(CKPT).cuda().eval()  # helper from demo.py

# 2) Load test input
images = torch.from_numpy(np.load(INP)["images"]).cuda()  # (1, 2, 3, 224, 224)
B, V = images.shape[:2]
intrinsics = torch.eye(3, device="cuda").expand(B, V, 3, 3).contiguous()

# 3) Per-stage timing — warmup + 5 reps, sync each side
timings = {}
for _ in range(3):  # warmup
    with torch.no_grad(): _ = encoder(images, intrinsics=intrinsics)
torch.cuda.synchronize()
for stage in ("backbone", "point_decoder", "gaussian_decoder",
              "camera_decoder", "gaussian_head", "rasterize"):
    t = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_stage(encoder, images, intrinsics, stage)  # cut at the relevant
        torch.cuda.synchronize(); t.append(time.perf_counter() - t0)
    timings[stage] = {"mean_ms": float(np.mean(t)*1000),
                      "p50_ms":  float(np.median(t)*1000)}

# 4) Final forward + rasterize → save artifacts
with torch.no_grad():
    out = encoder(images, intrinsics=intrinsics)

# 4a) Per-stage outputs (.npz schema matches our local dumps/per_block/)
np.savez(f"{OUT}/runpod_backbone_full.npz",
         out_0=out["backbone_tokens"].cpu().numpy(),
         out_3=out["dino_tokens"].cpu().numpy(),
         out_4=out["camera_pose"].cpu().numpy())

# 4b) Rendered RGB at the predicted poses
from src.model.decoder.decoder_splatting_gsplat import render_gaussians
rgb = render_gaussians(out["gaussians"], out["camera_pose"], H=224, W=224)
from PIL import Image
for v in range(V):
    Image.fromarray((rgb[0, v].cpu().numpy() * 255).clip(0,255).astype("uint8")) \
         .save(f"{OUT}/rendered_view{v}.png")

# 4c) Final .ply (loadable in SuperSplat)
from src.utils.gaussian_io import save_ply
save_ply(out["gaussians"], f"{OUT}/gaussians.ply")

# 4d) Timing
with open(f"{OUT}/timing.json", "w") as f:
    json.dump(timings, f, indent=2)
print("[runpod] done; artifacts in", OUT)
```

Run:

```bash
cd /workspace/yonosplat
python /workspace/runpod_ref.py 2>&1 | tee /workspace/ref/out/run.log
```

### 3.4 Download outputs back to local

```bash
# On local M1 ---------------------------------------------------------------
DEST=research/yonosplat_bootstrap/dumps/runpod_ref/
mkdir -p "$DEST"
scp 'pod:/workspace/ref/out/*' "$DEST/"
# Expect:
#   rendered_view0.png, rendered_view1.png  (~10 KB each)
#   gaussians.ply                            (~30 MB)
#   runpod_backbone_full.npz                 (~10 MB)
#   timing.json                              (<1 KB)
#   run.log                                  (text)
```

Then **stop the pod** (do not terminate — per `feedback_never_delete_pods`).

## 4. Cost estimate

| Item | Quantity | Unit (RunPod community A100 80 GB) | Subtotal |
|---|---:|---:|---:|
| Pod time (setup + run + download) | 30 min | ~$1.50 / h | ~$0.75 |
| Disk persistent (if reused) | 1 day x 40 GB | ~$0.20 / day | ~$0.20 |
| Bandwidth (~100 MB up + 100 MB down) | — | included | $0 |
| **Total (one validation pass)** | | | **~$1** |

Worst case (CUDA wheel build retries, two re-runs): **~$3**. No credit
burn-rate concerns at this scale.

## 5. What we compare locally after download

Once `dumps/runpod_ref/` exists, extend `meadow_sb/scripts/e2e_test.py`
with two extra reference channels:

| Local MLX output | RunPod ref | Tolerance |
|---|---|---|
| MLX-rendered RGB (Tier 2 Metal rasterizer) | `rendered_view*.png` | PSNR >= 25 dB |
| MLX `.ply` output | `gaussians.ply` | Gaussian count exact; centre-of-mass within 1e-2; mean opacity within 5% |
| MLX `backbone_full` | `runpod_backbone_full.npz` | `max |delta| < 1e-3` (CUDA cross-check) |

The CUDA-vs-CPU diff on the backbone is itself useful — it tells us how
much of any residual MLX-vs-PT delta is "MLX bug" vs "CUDA matmul reorder
noise".

## 6. What we deliberately do NOT do on RunPod

- **No training.** Forward-only validation. YoNoSplat MLX port is
  inference-only.
- **No bench against other splat libraries.** Out of scope.
- **No persistent pod.** Spin up, run, download, **stop** (never terminate).

## 7. When to actually trigger this

Run the RunPod pass only when **all** of the following are true:

1. Agents A-G have landed (modules in `meadow_sb/models/` + a weight
   converter producing MLX npz from `re10k.ckpt`).
2. Local `python meadow_sb/scripts/e2e_test.py` shows backbone diff
   `max |delta| < 1e-3` against the PyTorch-CPU dumps in
   `dumps/per_block/backbone_full.npz`. If we fail the CPU gate, RunPod
   won't help — it's an MLX correctness bug.
3. We have a working CPU-fallback or Tier-1 `gsplat` rasterizer locally so
   there is *some* MLX-side rendered image to compare against the RunPod
   `rendered_view*.png`.

Until then, this doc is a checklist; don't run it.

## 8. Outputs returned to local

| File | Size | Used by |
|---|---:|---|
| `rendered_view0.png`, `rendered_view1.png` | ~10 KB | Tier 2 rasterizer PSNR gate |
| `gaussians.ply` | ~30 MB | `.ply` round-trip + SuperSplat sanity |
| `runpod_backbone_full.npz` | ~10 MB | CUDA-vs-CPU sanity on encoder output |
| `timing.json` | <1 KB | A100 baseline for the "M1 vs A100" speed table |
| `run.log` | <100 KB | Forensics if any step fails |

## 9. Integration boundary note (MLX <-> CUDA rasterizer)

When we later compare an MLX-side Gaussian struct against the RunPod-side
struct, the conversion path is:

```
MLX Gaussians (mx.array fields)
  → np.array via mx.eval + np.array(...)
  → torch.from_numpy(...).cuda() at the rasterizer boundary
  → diff_gaussian_rasterization → image
```

This is fine for an *offline* PSNR check. It is **not** a path we ever
plan to ship — the Tier-2 Metal rasterizer (Agent F) replaces the CUDA
extension entirely on Apple hardware. The RunPod pass exists to *score*
that Metal rasterizer, not to be part of it.
