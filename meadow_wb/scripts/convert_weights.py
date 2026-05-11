"""Convert SAM 3D Objects PyTorch checkpoints -> MLX npz files per module.

Source: https://huggingface.co/facebook/sam-3d-objects (gated; user must
request access).

Pipeline checkpoints we convert (Gaussian-splat-only path, mesh decoder
intentionally skipped per `meadow_wb/docs/PORT_PLAN.md` scope cuts):

    HF checkpoint               -> npz
    ---------------------------    -----------------------
    ss_generator.ckpt           -> ss_flow.npz       (Stage-1 flow DiT, ~6.7 GB src)
                                   ss_embedder.npz   (DINO + fuser, image cond)
    slat_generator.ckpt         -> slat_flow.npz     (Stage-2 latent flow DiT, ~4.9 GB src)
                                   slat_embedder.npz (DINO + fuser, image cond)
    ss_decoder.ckpt             -> ss_decoder.npz    (occupancy decoder, ~150 MB)
    slat_decoder_gs.ckpt        -> slat_decoder_gs.npz       (gaussian splat decoder, ~170 MB)
    slat_decoder_gs_4.ckpt      -> slat_decoder_gs_4.npz     (high-res gs decoder, ~170 MB)
    ss_encoder.safetensors      -> ss_encoder.npz    (small; only for re-encoding)

Source-side state_dict layout (verified by reading
`sam3d_objects/pipeline/inference_pipeline.py`):

* `ss_generator.ckpt` and `slat_generator.ckpt` are PyTorch-Lightning ckpts:
  - top-level dict has key `state_dict`
  - inside, every weight is prefixed with either:
        `_base_models.generator.`         <- CFG-wrapped TdfyWrapper backbone
        `_base_models.condition_embedder.`<- DINO + EmbedderFuser
  - We split those two prefixes into separate npz files (different MLX
    modules will load each).
* `ss_decoder.ckpt`, `slat_decoder_gs*.ckpt`, `slat_decoder_mesh.ckpt` are
  raw state_dicts (no `state_dict` wrapper, no prefix). Each maps directly
  onto its model class.
* `ss_encoder.safetensors` is a safetensors file with bare module-name keys.

Tensor transforms:
- Conv2d weight (4D)  : (out, in, kH, kW)        -> (out, kH, kW, in)         [MLX channels-last]
- Conv3d weight (5D)  : (out, in, kD, kH, kW)    -> (out, kD, kH, kW, in)     [MLX channels-last]
- Linear / LayerNorm / RMSNorm / pos-embed: kept verbatim
- Integer buffers: kept as int64; everything else cast to float32

Usage:
    .venv/bin/python meadow_wb/weights/convert.py \
        --ckpt-dir checkpoints/hf/checkpoints \
        --out     meadow_wb/weights/sam3d_objects/

    # inspect-only (no npz produced; just dump per-key shapes/dtypes):
    .venv/bin/python meadow_wb/weights/convert.py \
        --ckpt-dir checkpoints/hf/checkpoints \
        --inspect

The `--inspect` flag is useful before access to the gated HF repo is
granted: once `ckpt-dir` is populated we can verify our prefix assumptions
match the actual ckpt without producing any output.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import OrderedDict, defaultdict
from typing import Callable, Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared tensor transforms
# ---------------------------------------------------------------------------

# Integer buffers we deliberately keep as int64 (everything else -> float32).
_INT_KEY_PATTERNS = (
    re.compile(r"\.(faces|indices|coords?)$"),
)


def _is_int_buffer(key: str, tensor: torch.Tensor) -> bool:
    if tensor.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool):
        return True
    return any(p.search(key) for p in _INT_KEY_PATTERNS)


def convert_tensor(key: str, tensor: torch.Tensor) -> np.ndarray:
    """PyTorch tensor -> numpy with MLX-friendly layout/dtype."""
    if _is_int_buffer(key, tensor):
        arr = tensor.detach().cpu().to(torch.int64).numpy()
    else:
        arr = tensor.detach().cpu().to(torch.float32).numpy()

    # Conv2d weight: (out, in, kH, kW) -> (out, kH, kW, in)
    if arr.ndim == 4 and _looks_like_conv2d(key):
        arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))

    # Conv3d weight: (out, in, kD, kH, kW) -> (out, kD, kH, kW, in)
    if arr.ndim == 5 and _looks_like_conv3d(key):
        arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 4, 1))

    return arr


# Heuristic: a 4D weight tensor that lives under a *.weight key and not
# inside something obviously transformer-shaped is a Conv2d. The DINO ViT
# patch_embed.proj.weight is the only Conv2d we routinely hit on the
# embedder side. The flow models themselves are pure Linear so 4D tensors
# there would be unexpected.
def _looks_like_conv2d(key: str) -> bool:
    return key.endswith(".weight") and (
        "patch_embed" in key
        or ".conv" in key
        or "patch_proj" in key
    )


def _looks_like_conv3d(key: str) -> bool:
    # SS encoder/decoder use Conv3d throughout; everything 5D is Conv3d.
    return key.endswith(".weight")


# ---------------------------------------------------------------------------
# Per-checkpoint plan
# ---------------------------------------------------------------------------

def _strip_prefix(prefix: str) -> Callable[[str], str | None]:
    """Return a fn that strips `prefix` from a key, or returns None if absent."""
    n = len(prefix)
    def fn(key: str) -> str | None:
        return key[n:] if key.startswith(prefix) else None
    return fn


# Each entry: (ckpt_filename, [(out_npz_name, key_filter_fn), ...], state_dict_key)
# state_dict_key=None means the ckpt IS the state_dict.
PLAN = [
    (
        "ss_generator.ckpt",
        [
            ("ss_flow",     _strip_prefix("_base_models.generator.")),
            ("ss_embedder", _strip_prefix("_base_models.condition_embedder.")),
        ],
        "state_dict",
    ),
    (
        "slat_generator.ckpt",
        [
            ("slat_flow",     _strip_prefix("_base_models.generator.")),
            ("slat_embedder", _strip_prefix("_base_models.condition_embedder.")),
        ],
        "state_dict",
    ),
    (
        "ss_decoder.ckpt",
        [
            ("ss_decoder", lambda k: k),  # all keys
        ],
        None,
    ),
    (
        "slat_decoder_gs.ckpt",
        [
            ("slat_decoder_gs", lambda k: k),
        ],
        None,
    ),
    (
        "slat_decoder_gs_4.ckpt",
        [
            ("slat_decoder_gs_4", lambda k: k),
        ],
        None,
    ),
    # ss_encoder is small + safetensors; convert separately if user wants it.
    (
        "ss_encoder.safetensors",
        [
            ("ss_encoder", lambda k: k),
        ],
        None,
    ),
    # slat_decoder_mesh.ckpt intentionally skipped (Gaussian-splat-only path).
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_ckpt(path: str, state_dict_key: str | None) -> dict:
    """Load a checkpoint, returning a flat name -> tensor dict.

    Handles three formats:
    1. .ckpt with a `state_dict` wrapper (Lightning).
    2. .ckpt with the state_dict at top level.
    3. .safetensors (lazy-loaded via safetensors.torch.load_file).
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if state_dict_key is not None and isinstance(obj, dict) and state_dict_key in obj:
        sd = obj[state_dict_key]
    else:
        sd = obj
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unexpected ckpt structure: {type(sd).__name__}")
    return sd


# ---------------------------------------------------------------------------
# Inspection (no output written)
# ---------------------------------------------------------------------------

def _summarise_state_dict(sd: dict) -> tuple[int, int, dict, dict]:
    """Return (n_tensors, n_params, prefix_count, dtype_count)."""
    n_tensors = 0
    n_params = 0
    prefix_count: dict[str, int] = defaultdict(int)
    dtype_count: dict[str, int] = defaultdict(int)
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        n_tensors += 1
        n_params += int(np.prod(v.shape))
        head = k.split(".", 2)
        prefix = ".".join(head[:2]) if len(head) > 1 else head[0]
        prefix_count[prefix] += 1
        dtype_count[str(v.dtype)] += 1
    return n_tensors, n_params, dict(prefix_count), dict(dtype_count)


def inspect(ckpt_dir: str) -> None:
    print(f"[inspect] scanning ckpt dir: {ckpt_dir}\n")
    grand_total_tensors = 0
    grand_total_params = 0
    for fname, _routes, sd_key in PLAN:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {fname}")
            continue
        size = os.path.getsize(path)
        try:
            sd = _load_ckpt(path, sd_key)
        except Exception as e:
            print(f"  FAILED to load {fname}: {e}")
            continue
        n_t, n_p, pref, dts = _summarise_state_dict(sd)
        grand_total_tensors += n_t
        grand_total_params += n_p
        print(f"== {fname} ({size/1e9:.3f} GB on disk) ==")
        print(f"   tensors={n_t}  params={n_p/1e6:.2f} M  dtypes={dts}")
        print(f"   top-2 prefixes:")
        for p, c in sorted(pref.items(), key=lambda x: -x[1])[:12]:
            print(f"      {c:5d}  {p}")
        print()
    print(f"GRAND TOTAL: {grand_total_tensors} tensors, "
          f"{grand_total_params/1e6:.1f} M params")


# ---------------------------------------------------------------------------
# Convert driver
# ---------------------------------------------------------------------------

def convert_all(ckpt_dir: str, out_dir: str, no_compress: bool = False) -> None:
    keymap_rows: list[tuple[str, str, str, list[int], str]] = []
    file_summary: list[tuple[str, int, int]] = []
    grand_orphans: dict[str, list[str]] = {}

    # Don't create the output dir until we actually have something to write,
    # so that "no checkpoints found" doesn't clobber a planning KEY_MAP.md.
    out_dir_created = False

    for fname, routes, sd_key in PLAN:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"[convert] SKIP missing: {fname}")
            continue
        print(f"[convert] loading {fname} ...")
        sd = _load_ckpt(path, sd_key)
        print(f"[convert]   {len(sd)} top-level keys")

        # Bucket keys by route. Each key goes into the FIRST matching route.
        buckets: dict[str, dict[str, str]] = {name: {} for name, _ in routes}
        orphans: list[str] = []
        for raw_key in sd.keys():
            placed = False
            for route_name, key_fn in routes:
                new_key = key_fn(raw_key)
                if new_key is None:
                    continue
                buckets[route_name][raw_key] = new_key
                placed = True
                break
            if not placed:
                orphans.append(raw_key)
        if orphans:
            grand_orphans[fname] = orphans

        # Write each bucket.
        for route_name, raw_to_new in buckets.items():
            if not raw_to_new:
                print(f"[convert]   {route_name}: 0 keys, skipping")
                continue
            if not out_dir_created:
                os.makedirs(out_dir, exist_ok=True)
                out_dir_created = True
            out_path = os.path.join(out_dir, f"{route_name}.npz")
            out_dict: dict[str, np.ndarray] = {}
            for raw_key in sorted(raw_to_new):
                new_key = raw_to_new[raw_key]
                arr = convert_tensor(raw_key, sd[raw_key])
                if new_key in out_dict:
                    raise RuntimeError(
                        f"Key collision in {route_name}: "
                        f"{raw_key} -> {new_key}"
                    )
                out_dict[new_key] = arr
                keymap_rows.append((route_name, raw_key, new_key, list(arr.shape), str(arr.dtype)))

            if no_compress:
                np.savez(out_path, **out_dict)
            else:
                np.savez_compressed(out_path, **out_dict)
            sz = os.path.getsize(out_path)
            file_summary.append((route_name, len(out_dict), sz))
            print(f"[convert]   {route_name}.npz tensors={len(out_dict):4d} size={sz/1e6:8.2f} MB")
        del sd  # free memory between checkpoints (largest is ~6.7 GB)

    if not file_summary:
        print("[convert] no checkpoints found; nothing written. "
              "(Planning KEY_MAP.md in meadow_wb/weights/ left intact.)")
        return

    # KEY_MAP.md
    kmap_path = os.path.join(out_dir, "KEY_MAP.md")
    with open(kmap_path, "w") as f:
        f.write("# SAM 3D Objects weight key map (PyTorch -> MLX)\n\n")
        f.write("Generated by `meadow_wb/weights/convert.py`.\n\n")
        f.write("## File summary\n\n")
        f.write("| file | tensors | size |\n|---|---|---|\n")
        total_size = 0
        total_tensors = 0
        for name, n, sz in file_summary:
            f.write(f"| `{name}.npz` | {n} | {sz/1e6:.2f} MB |\n")
            total_size += sz
            total_tensors += n
        f.write(f"| **total** | **{total_tensors}** | **{total_size/1e6:.2f} MB** |\n\n")
        f.write("## Tensor layout conventions\n\n")
        f.write("- Conv2d weights `(out, in, kH, kW)` -> MLX channels-last "
                "`(out, kH, kW, in)`.\n")
        f.write("- Conv3d weights `(out, in, kD, kH, kW)` -> MLX channels-last "
                "`(out, kD, kH, kW, in)`.\n")
        f.write("- Linear / LayerNorm / RMSNorm / position-embedding tensors "
                "are kept verbatim.\n")
        f.write("- Integer buffers (`*.faces`, `*.indices`, `*.coords`) stay "
                "int64; everything else is float32.\n\n")
        if grand_orphans:
            f.write("## Orphan keys (NOT WRITTEN to any npz)\n\n")
            for fname, keys in grand_orphans.items():
                f.write(f"### From `{fname}`\n\n")
                for k in keys:
                    f.write(f"- `{k}`\n")
                f.write("\n")
        # Sample mapping table (head per group, not every key)
        f.write("## Key mapping samples (first 20 per group)\n\n")
        f.write("| group | pt_key | mx_key | shape | dtype |\n|---|---|---|---|---|\n")
        per_group: dict[str, int] = defaultdict(int)
        for g, p, m, s, d in keymap_rows:
            if per_group[g] >= 20:
                continue
            per_group[g] += 1
            f.write(f"| {g} | `{p}` | `{m}` | {s} | {d} |\n")
    print(f"[convert] wrote {kmap_path}")

    grand_size = sum(sz for _, _, sz in file_summary)
    print(f"[convert] DONE. total {sum(n for _, n, _ in file_summary)} tensors, "
          f"{grand_size/1e6:.2f} MB on disk")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="Directory holding HF ckpts (must contain ss_generator.ckpt etc.)")
    ap.add_argument("--out", default="meadow_wb/weights/sam3d_objects/", help="Output dir for npz files")
    ap.add_argument("--no-compress", action="store_true", help="Use np.savez (faster, larger)")
    ap.add_argument("--inspect", action="store_true", help="Just print shapes/prefixes; do not write npz")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.ckpt_dir)
    else:
        convert_all(args.ckpt_dir, args.out, no_compress=args.no_compress)


if __name__ == "__main__":
    main()
