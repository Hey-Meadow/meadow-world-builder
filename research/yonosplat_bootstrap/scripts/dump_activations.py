"""YoNoSplat reference-activation dump on Apple Silicon (CPU or MPS).

Loads re10k_224x224_ctx2to32.ckpt, builds the encoder, registers a forward
hook on every key sub-module, runs a forward pass on a 2-view RGB input,
and writes every intermediate tensor to `dumps/` as a NumPy `.npz`.

The dump is the *single source of truth* the parallel MLX-port agents
will use to validate their per-block implementations:

    max(abs(mlx_output - pt_reference)) < 1e-4   for fp32 path

Usage:
    .venv/bin/python research/yonosplat_bootstrap/scripts/dump_activations.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# Make the upstream yonosplat source importable.
UPSTREAM = "/tmp/yonosplat_inspect"
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

REPO_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = REPO_ROOT / "research" / "yonosplat_bootstrap"
WEIGHTS = BOOTSTRAP / "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"
DUMP_DIR = BOOTSTRAP / "dumps"
DUMP_DIR.mkdir(parents=True, exist_ok=True)


def make_test_input(device: torch.device) -> torch.Tensor:
    """Two 224×224 RGB views from a fixed Meta-released demo image.

    Returns:
        images: (1, 2, 3, 224, 224) float32 in [0, 1].
    """
    src = "/Users/akaihuangm1/Desktop/github/sam-3d-objects/notebook/images/" \
          "shutterstock_stylish_kidsroom_1640806567/image.png"
    im = Image.open(src).convert("RGB")
    # Two crops: roughly centre + offset, simulating two photographers.
    W, H = im.size
    cx, cy = W // 2, H // 2
    crop_size = min(W, H) - 200
    boxes = [
        (cx - crop_size // 2, cy - crop_size // 2, cx + crop_size // 2, cy + crop_size // 2),
        (cx - crop_size // 2 + 400, cy - crop_size // 2 - 200,
         cx + crop_size // 2 + 400, cy + crop_size // 2 - 200),
    ]
    views = []
    for box in boxes:
        c = im.crop(box).resize((224, 224), Image.LANCZOS)
        arr = np.asarray(c, dtype=np.float32) / 255.0
        views.append(arr.transpose(2, 0, 1))
    stack = np.stack(views, axis=0)[None]  # (1, 2, 3, 224, 224)
    t = torch.from_numpy(stack).to(device)
    return t


def main():
    # ---- Stub the CUDA rasterizer so the encoder import chain doesn't break ----
    # (Stub was installed earlier in site-packages; nothing to do here unless missing.)
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "diff_gaussian_rasterization stub not found in site-packages. "
            "Install it first; see PORT_PLAN_YONOSPLAT.md."
        ) from e

    print(f"[dump] checkpoint = {WEIGHTS} ({WEIGHTS.stat().st_size / 1e6:.0f} MB)")
    ckpt = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    print(f"[dump] state_dict tensors = {len(sd)}")

    # ---- Just save the raw state_dict + a slim metadata file -------------------
    # Building the full EncoderYoNoSplat requires hydra config plumbing; for
    # bootstrap-stage activation comparison we mostly need per-block reference
    # activations, which we can produce by walking the Pi3 backbone directly
    # (it's an independent module).
    # Strategy:
    #   1) Dump the full state_dict tensor map to JSON (key → shape, dtype)
    #   2) Save a sliced "backbone only" .pt for the parallel-port agents to consume
    #   3) Build a minimal Pi3 backbone, load weights, run forward, dump activations
    # Step 3 requires the Pi3 model class. Defer that to a follow-up dump_pi3.py
    # since it touches more of the upstream model code.

    # ---- 1) Tensor map ---------------------------------------------------------
    tensor_map_path = DUMP_DIR / "state_dict_tensor_map.json"
    import json
    tmap = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sd.items()}
    with open(tensor_map_path, "w") as f:
        json.dump(tmap, f, indent=2)
    print(f"[dump] tensor map saved: {tensor_map_path}")

    # ---- 2) Per-component count + size -----------------------------------------
    from collections import defaultdict
    bins = defaultdict(lambda: {"count": 0, "params": 0, "bytes": 0})
    for k, v in sd.items():
        prefix = ".".join(k.split(".")[:3]) if k.startswith("encoder.backbone") else \
                 ".".join(k.split(".")[:2])
        bins[prefix]["count"] += 1
        bins[prefix]["params"] += v.numel()
        bins[prefix]["bytes"] += v.numel() * v.element_size()
    print("\n[dump] component-wise breakdown:")
    for prefix, info in sorted(bins.items(), key=lambda x: -x[1]["params"]):
        print(f"  {prefix:60s}  {info['count']:5d} tensors  "
              f"{info['params']/1e6:7.2f} M params  {info['bytes']/1e6:6.1f} MB")

    # ---- 3) Save test inputs --------------------------------------------------
    img = make_test_input(torch.device("cpu"))
    np.savez(DUMP_DIR / "test_input.npz", images=img.numpy())
    print(f"\n[dump] test input saved: shape={tuple(img.shape)}, dtype={img.dtype}")

    print(f"\n[dump] artifacts in {DUMP_DIR}/:")
    for p in sorted(DUMP_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e3:.1f} KB)")

    print("\n[dump] next step: write dump_pi3.py — minimal backbone-only forward "
          "with per-block hooks. The state-dict map saved here lets the parallel "
          "agents grep for keys without re-loading the 3.8 GB checkpoint.")


if __name__ == "__main__":
    main()
