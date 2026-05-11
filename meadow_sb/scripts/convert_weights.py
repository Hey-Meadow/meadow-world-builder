"""Convert YoNoSplat PyTorch checkpoint -> MLX npz files per module.

Source checkpoint (3.86 GB, 1222 tensors, 965 M params):
    research/yonosplat_bootstrap/weights/yonosplat/re10k_224x224_ctx2to32.ckpt

Top-level prefix layout (verified against
`research/yonosplat_bootstrap/dumps/state_dict_tensor_map.json`):

    pt-prefix                                                  -> npz                     count   params
    --------------------------------------------------------------------------------------------------
    encoder.backbone.encoder.*                                 -> dinov2_encoder.npz        343   304.37 M
    encoder.backbone.decoder.*                                 -> croco_decoder.npz         648   453.55 M
    encoder.point_decoder.*                                    -> point_decoder.npz          64    66.13 M
    encoder.gaussian_decoder.*                                 -> gaussian_decoder.npz       64    66.13 M
    encoder.camera_decoder.*                                   -> camera_decoder.npz         64    65.60 M
    encoder.{gaussian_head,point_head,camera_head}.*           -> heads.npz                  32     4.17 M
    + encoder.backbone.intrinsic_head.* + encoder.rgb_embed.*
    encoder.backbone.intrinsics_embed_layer.*                  -> intrinsics_embed.npz        4     5.02 M
    encoder.backbone.register_token + image_mean + image_std   -> register_token.npz          3     0.01 M
    --------------------------------------------------------------------------------------------------
    total                                                                                  1222   965.01 M

Tensor transforms:
- Conv2d weight (4D)  : (out, in, kH, kW) -> (out, kH, kW, in)  [MLX channels-last]
  Triggered by: `patch_embed.proj.weight`, `intrinsics_embed_layer.proj.weight`,
  `rgb_embed.proj.weight`. NOT applied to other 4D tensors (register_token,
  image_mean, image_std) which are buffers / tokens with no spatial meaning.
- Linear / LayerNorm / RMSNorm / position embeddings: kept verbatim.
- Integer buffers (names ending in `.coord`, `.coords`, `.indices`, `.sizes`,
  or already integer dtype): kept as int32/int64.
- Everything else cast to float32.

Usage:
    .venv/bin/python -m meadow_sb.scripts.convert_weights \
        --ckpt research/yonosplat_bootstrap/weights/yonosplat/re10k_224x224_ctx2to32.ckpt \
        --out  meadow_sb/weights/

    # Inspect-only: prints key counts/shapes per group, no npz written.
    .venv/bin/python -m meadow_sb.scripts.convert_weights \
        --ckpt research/yonosplat_bootstrap/weights/yonosplat/re10k_224x224_ctx2to32.ckpt \
        --inspect
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Group plan
# ---------------------------------------------------------------------------

# Each entry: (out_npz_name, filter_fn). Filter functions are applied in order;
# the FIRST match wins, so list more-specific filters before more-general ones.
# (Here all filters are mutually exclusive by construction.)
GROUP_PLAN: list[tuple[str, Callable[[str], bool]]] = [
    # Specific sub-prefixes of encoder.backbone.* first.
    ("dinov2_encoder",   lambda k: k.startswith("encoder.backbone.encoder.")),
    ("croco_decoder",    lambda k: k.startswith("encoder.backbone.decoder.")),
    ("intrinsics_embed", lambda k: k.startswith("encoder.backbone.intrinsics_embed_layer.")),
    ("register_token",   lambda k: k in (
        "encoder.backbone.register_token",
        "encoder.backbone.image_mean",
        "encoder.backbone.image_std",
    )),
    # Per-decoder buckets.
    ("point_decoder",    lambda k: k.startswith("encoder.point_decoder.")),
    ("gaussian_decoder", lambda k: k.startswith("encoder.gaussian_decoder.")),
    ("camera_decoder",   lambda k: k.startswith("encoder.camera_decoder.")),
    # Heads (mixed prefixes; this must come AFTER the more-specific
    # `encoder.backbone.intrinsics_embed_layer` filter above, otherwise the
    # backbone.intrinsic_head match would be ambiguous — but intrinsic_head
    # and intrinsics_embed_layer are different prefixes, so order is safe).
    ("heads",            lambda k: (
        k.startswith("encoder.gaussian_head.")
        or k.startswith("encoder.point_head.")
        or k.startswith("encoder.camera_head.")
        or k.startswith("encoder.backbone.intrinsic_head.")
        or k.startswith("encoder.rgb_embed.")
    )),
]

EXPECTED_COUNTS: dict[str, int] = {
    "dinov2_encoder":   343,
    "croco_decoder":    648,
    "point_decoder":     64,
    "gaussian_decoder":  64,
    "camera_decoder":    64,
    "heads":             32,
    "intrinsics_embed":   4,
    "register_token":     3,
}

# Conv2d weight keys to transpose (out, in, kH, kW) -> (out, kH, kW, in).
# Restricted to actual convs; other 4D tensors (register_token, image_mean,
# image_std) are *not* spatial and must be left alone.
_CONV2D_WEIGHT_PATTERNS = (
    re.compile(r"\.patch_embed\.proj\.weight$"),
    re.compile(r"^encoder\.backbone\.intrinsics_embed_layer\.proj\.weight$"),
    re.compile(r"^encoder\.rgb_embed\.proj\.weight$"),
)

# Integer-buffer key patterns (also: any tensor with int dtype is kept int).
_INT_KEY_PATTERNS = (
    re.compile(r"\.(coords?|indices|sizes)$"),
)


# ---------------------------------------------------------------------------
# Tensor transforms
# ---------------------------------------------------------------------------

def _is_conv2d_weight(key: str) -> bool:
    return any(p.search(key) for p in _CONV2D_WEIGHT_PATTERNS)


def _is_int_buffer(key: str, tensor: torch.Tensor) -> bool:
    if tensor.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool):
        return True
    return any(p.search(key) for p in _INT_KEY_PATTERNS)


def convert_tensor(key: str, tensor: torch.Tensor) -> np.ndarray:
    """PyTorch tensor -> numpy with MLX-friendly layout/dtype."""
    if _is_int_buffer(key, tensor):
        # Preserve signed-ness; int32 stays int32, others -> int64.
        if tensor.dtype == torch.int32:
            arr = tensor.detach().cpu().to(torch.int32).numpy()
        else:
            arr = tensor.detach().cpu().to(torch.int64).numpy()
    else:
        arr = tensor.detach().cpu().to(torch.float32).numpy()

    if arr.ndim == 4 and _is_conv2d_weight(key):
        # (out, in, kH, kW) -> (out, kH, kW, in)
        arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))

    return arr


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_state_dict(path: str) -> dict:
    """Load YoNoSplat .ckpt and return the flat tensor state_dict.

    The re10k checkpoint is a PyTorch-Lightning .ckpt with a top-level
    `state_dict` key; we unwrap it here.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise RuntimeError(f"Unexpected ckpt structure: {type(obj).__name__}")
    # Drop any non-tensor entries.
    return {k: v for k, v in sd.items() if hasattr(v, "shape") and hasattr(v, "dtype")}


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def bucket_keys(sd: dict) -> tuple[dict[str, list[str]], list[str]]:
    """Place every key into exactly one group; report orphans."""
    buckets: dict[str, list[str]] = {name: [] for name, _ in GROUP_PLAN}
    orphans: list[str] = []
    for key in sd.keys():
        placed = False
        for name, fn in GROUP_PLAN:
            if fn(key):
                buckets[name].append(key)
                placed = True
                break
        if not placed:
            orphans.append(key)
    return buckets, orphans


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------

def inspect(ckpt_path: str) -> None:
    print(f"[inspect] loading {ckpt_path} ...")
    size = os.path.getsize(ckpt_path)
    sd = load_state_dict(ckpt_path)
    print(f"[inspect] {len(sd)} tensors, ckpt size {size/1e9:.3f} GB on disk\n")

    buckets, orphans = bucket_keys(sd)

    total_keys = 0
    total_params = 0
    print(f"{'group':22s} {'keys':>5s} {'params(M)':>12s} {'expected':>10s} {'ok':>4s}")
    print("-" * 60)
    for name, _ in GROUP_PLAN:
        keys = buckets[name]
        n_params = sum(int(np.prod(sd[k].shape)) for k in keys)
        expected = EXPECTED_COUNTS.get(name, -1)
        ok = "OK" if expected == len(keys) else "FAIL"
        total_keys += len(keys)
        total_params += n_params
        print(f"{name:22s} {len(keys):5d} {n_params/1e6:12.2f} {expected:10d} {ok:>4s}")
    print("-" * 60)
    print(f"{'total':22s} {total_keys:5d} {total_params/1e6:12.2f}")
    print()

    if orphans:
        print(f"[inspect] ORPHANS ({len(orphans)}):")
        for k in orphans[:50]:
            print(f"   - {k}")
        if len(orphans) > 50:
            print(f"   ... and {len(orphans) - 50} more")
    else:
        print("[inspect] no orphan keys (all 1222 placed cleanly)")

    # Show a representative tensor from each group.
    print("\n[inspect] sample tensors per group:")
    for name, _ in GROUP_PLAN:
        keys = buckets[name]
        if not keys:
            continue
        k = sorted(keys)[0]
        v = sd[k]
        print(f"   {name:22s} {tuple(v.shape)!s:24s} {v.dtype!s:18s} {k}")


# ---------------------------------------------------------------------------
# Convert mode
# ---------------------------------------------------------------------------

def convert_all(ckpt_path: str, out_dir: str, no_compress: bool = False) -> dict[str, str]:
    print(f"[convert] loading {ckpt_path} ...")
    sd = load_state_dict(ckpt_path)
    print(f"[convert] {len(sd)} tensors loaded")

    buckets, orphans = bucket_keys(sd)
    if orphans:
        raise RuntimeError(
            f"refusing to convert: {len(orphans)} orphan keys; first 5: "
            f"{orphans[:5]}"
        )

    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}
    conv2d_hits: list[str] = []

    for name, _ in GROUP_PLAN:
        keys = buckets[name]
        if not keys:
            print(f"[convert]   {name}: 0 keys, skipping")
            continue
        out_dict: dict[str, np.ndarray] = {}
        for raw_key in sorted(keys):
            arr = convert_tensor(raw_key, sd[raw_key])
            if _is_conv2d_weight(raw_key):
                conv2d_hits.append(raw_key)
            out_dict[raw_key] = arr
        out_path = os.path.join(out_dir, f"{name}.npz")
        if no_compress:
            np.savez(out_path, **out_dict)
        else:
            np.savez_compressed(out_path, **out_dict)
        sz = os.path.getsize(out_path)
        written[name] = out_path
        print(f"[convert]   {name}.npz tensors={len(out_dict):4d} size={sz/1e6:8.2f} MB")

    print(f"[convert] Conv2d transpose applied to: {conv2d_hits}")
    print(f"[convert] DONE. {len(written)} npz files in {out_dir}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="YoNoSplat PT -> MLX npz converter")
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Path to re10k_224x224_ctx2to32.ckpt",
    )
    ap.add_argument(
        "--out",
        default="meadow_sb/weights/",
        help="Output directory for npz files",
    )
    ap.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed (faster but larger)",
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="Print group counts/shapes without writing any npz",
    )
    args = ap.parse_args()

    if args.inspect:
        inspect(args.ckpt)
    else:
        convert_all(args.ckpt, args.out, no_compress=args.no_compress)


if __name__ == "__main__":
    main()
