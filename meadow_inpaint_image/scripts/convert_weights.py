"""Convert big-lama PyTorch-Lightning ckpt -> flat MLX npz.

Usage:
  python3.11 scripts/convert_weights.py [--ckpt PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import types
from pathlib import Path

import numpy as np


def stub_pl():
    """Install minimal pytorch_lightning stubs so torch.load can unpickle."""
    def _stub(name):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        return m

    class _Generic:
        def __init__(self, *a, **kw):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    if "pytorch_lightning" not in sys.modules:
        _stub("pytorch_lightning")
        _stub("pytorch_lightning.callbacks")
        cb = _stub("pytorch_lightning.callbacks.model_checkpoint")
        cb.ModelCheckpoint = _Generic
        _stub("pytorch_lightning.utilities")
        _stub("pytorch_lightning.utilities.cloud_io")
        _stub("pytorch_lightning.core")
        core_l = _stub("pytorch_lightning.core.lightning")
        core_l.LightningModule = _Generic


# --- name mapping --------------------------------------------------------------

# Helpers: take a PT key starting with 'generator.model.{N}.' and return the MLX flat key,
# or None if it's a buffer we drop (num_batches_tracked).

def map_pt_to_mlx(pt_key: str):
    """Return (mlx_key, transpose_kind) or (None, None) to drop."""
    if not pt_key.startswith("generator.model."):
        return None, None
    tail = pt_key[len("generator.model."):]
    parts = tail.split(".")
    idx = int(parts[0])
    rest = parts[1:]

    # drop tracker buffers
    if rest and rest[-1] == "num_batches_tracked":
        return None, None

    # Determine module prefix
    if idx == 1:
        prefix = "b1"
    elif idx in (2, 3, 4):
        prefix = f"downs.{idx - 2}"
    elif 5 <= idx <= 22:
        prefix = f"res.{idx - 5}"
    elif idx == 24:
        prefix = "ups_ct.0"
    elif idx == 25:
        prefix = "ups_bn.0"
    elif idx == 27:
        prefix = "ups_ct.1"
    elif idx == 28:
        prefix = "ups_bn.1"
    elif idx == 30:
        prefix = "ups_ct.2"
    elif idx == 31:
        prefix = "ups_bn.2"
    elif idx == 34:
        prefix = "final"
    else:
        return None, None

    # Map sub-path.
    # For BatchNorm-like idx (25, 28, 31), `rest` is e.g. ['weight'], ['running_mean'], etc.
    if idx in (25, 28, 31):
        # ups_bn.X
        return f"{prefix}.{'.'.join(rest)}", "bn"

    if idx in (24, 27, 30):
        # ups_ct.X (ConvTranspose2d): rest is ['weight'] or ['bias']
        return f"{prefix}.{'.'.join(rest)}", ("convtranspose" if rest[-1] == "weight" else "passthrough")

    if idx == 34:
        # final conv (k=7), rest is ['weight'] or ['bias']
        return f"{prefix}.{'.'.join(rest)}", ("conv2d" if rest[-1] == "weight" else "passthrough")

    # idx in {1, 2, 3, 4} or 5..22 -> FFC_BN_ACT or FFCResnetBlock branches
    # Sub-paths look like:
    #   ffc.convl2l.weight                       -> ffc.convl2l.conv.weight   (conv2d transpose)
    #   ffc.convl2g.weight                       -> ffc.convl2g.conv.weight   (conv2d transpose)
    #   ffc.convg2l.weight                       -> ffc.convg2l.conv.weight   (conv2d transpose)
    #   ffc.convg2g.conv1.0.weight               -> ffc.convg2g.conv1_conv.weight (conv2d)
    #   ffc.convg2g.conv1.1.{weight,bias,running_*} -> ffc.convg2g.conv1_bn.{...} (bn)
    #   ffc.convg2g.fu.conv_layer.weight         -> ffc.convg2g.fu.conv_layer.weight (conv2d)
    #   ffc.convg2g.fu.bn.{weight,bias,running_*} -> ffc.convg2g.fu.bn.{...} (bn)
    #   ffc.convg2g.conv2.weight                 -> ffc.convg2g.conv2.weight (conv2d)
    #   bn_l.{weight,bias,running_*}             -> bn_l.{...}
    #   bn_g.{weight,bias,running_*}             -> bn_g.{...}
    # And for ResnetBlock, an extra prefix "conv1." or "conv2." is present before the above.

    # Identify possible conv1./conv2. prefix
    sub_prefix = ""
    if rest and rest[0] in ("conv1", "conv2") and idx >= 5:  # only resblocks have this
        sub_prefix = rest[0] + "."
        rest = rest[1:]

    if not rest:
        return None, None

    head = rest[0]
    leaf = rest[-1]

    if head == "ffc":
        # rest like ['ffc', 'convl2l', 'weight'] or ['ffc', 'convg2g', 'conv1', '0', 'weight'] etc.
        sub = rest[1]
        if sub in ("convl2l", "convl2g", "convg2l"):
            assert leaf == "weight"
            mlx_path = f"{prefix}.{sub_prefix}ffc.{sub}.conv.weight"
            return mlx_path, "conv2d"
        elif sub == "convg2g":
            # SpectralTransform path
            inner = rest[2:]
            if inner[0] == "conv1":
                # conv1.0.weight  or  conv1.1.{weight,bias,running_mean,running_var}
                if inner[1] == "0":
                    mlx_path = f"{prefix}.{sub_prefix}ffc.convg2g.conv1_conv.{inner[-1]}"
                    kind = "conv2d" if inner[-1] == "weight" else "passthrough"
                    return mlx_path, kind
                elif inner[1] == "1":
                    mlx_path = f"{prefix}.{sub_prefix}ffc.convg2g.conv1_bn.{inner[-1]}"
                    return mlx_path, "bn"
            elif inner[0] == "fu":
                # fu.conv_layer.weight  or  fu.bn.{..}
                if inner[1] == "conv_layer":
                    mlx_path = f"{prefix}.{sub_prefix}ffc.convg2g.fu.conv_layer.weight"
                    return mlx_path, "conv2d"
                elif inner[1] == "bn":
                    mlx_path = f"{prefix}.{sub_prefix}ffc.convg2g.fu.bn.{inner[-1]}"
                    return mlx_path, "bn"
            elif inner[0] == "conv2":
                mlx_path = f"{prefix}.{sub_prefix}ffc.convg2g.conv2.weight"
                return mlx_path, "conv2d"
        return None, None
    elif head in ("bn_l", "bn_g"):
        # rest like ['bn_l', 'weight']
        mlx_path = f"{prefix}.{sub_prefix}{head}.{leaf}"
        return mlx_path, "bn"
    elif head == "weight" and idx == 34:
        # already handled above
        return f"{prefix}.weight", "conv2d"
    else:
        return None, None


def transpose_for_mlx(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "conv2d":
        # PT (out, in, kH, kW) -> MLX (out, kH, kW, in)
        assert arr.ndim == 4, f"conv2d expected 4D got {arr.shape}"
        return np.transpose(arr, (0, 2, 3, 1)).astype(np.float32)
    if kind == "convtranspose":
        # PT (in, out, kH, kW) -> MLX (out, kH, kW, in)
        assert arr.ndim == 4, f"convtranspose expected 4D got {arr.shape}"
        return np.transpose(arr, (1, 2, 3, 0)).astype(np.float32)
    # bn or passthrough → ensure fp32, keep shape
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--ckpt", default=str(here / "weights" / "big-lama" / "models" / "best.ckpt"))
    ap.add_argument("--out", default=str(here / "weights" / "lama_mlx.npz"))
    args = ap.parse_args()

    stub_pl()
    import torch

    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    print(f"state_dict entries: {len(sd)}")

    out_dict = {}
    dropped = []
    unmapped = []
    for k, v in sd.items():
        if not k.startswith("generator."):
            continue
        if k.endswith(".num_batches_tracked"):
            dropped.append(k)
            continue
        mlx_key, kind = map_pt_to_mlx(k)
        if mlx_key is None:
            unmapped.append(k)
            continue
        arr = v.detach().cpu().numpy()
        out_arr = transpose_for_mlx(arr, kind)
        out_dict[mlx_key] = out_arr

    print(f"Mapped tensors: {len(out_dict)}")
    print(f"Dropped (num_batches_tracked): {len(dropped)}")
    if unmapped:
        print(f"UNMAPPED ({len(unmapped)}):")
        for k in unmapped[:20]:
            print("  ", k)

    total_params = sum(int(np.prod(a.shape)) for a in out_dict.values())
    print(f"Total params: {total_params:,}  ({total_params/1e6:.1f} M)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    sz = out_path.stat().st_size / (1024 ** 2)
    print(f"Saved {out_path}  ({sz:.1f} MiB)")


if __name__ == "__main__":
    main()
