"""Convert ProPainter PyTorch .pth weights to MLX-friendly .npz files.

Produces three npz files:
  - raft.npz          (RAFT optical flow)
  - rfc.npz           (RecurrentFlowCompletion)
  - propainter_main.npz (main inpainter)

Conversion rules
----------------
* Conv2d weight (Cout, Cin, kH, kW) -> (Cout, kH, kW, Cin) (MLX OHWI layout)
* Conv3d weight (Cout, Cin, kT, kH, kW) -> (Cout, kT, kH, kW, Cin) (MLX ODHWI)
* Linear weight unchanged
* `module.` prefix stripped from RAFT keys (DataParallel artefact)
* BatchNorm / GroupNorm tensors copied verbatim
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).numpy()


def _convert_conv2d_weight(w: np.ndarray) -> np.ndarray:
    # PT (Cout, Cin, kH, kW) -> MLX (Cout, kH, kW, Cin)
    return np.transpose(w, (0, 2, 3, 1))


def _convert_conv3d_weight(w: np.ndarray) -> np.ndarray:
    # PT (Cout, Cin, kT, kH, kW) -> MLX (Cout, kT, kH, kW, Cin)
    return np.transpose(w, (0, 2, 3, 4, 1))


def _is_conv2d_weight(key: str, shape: tuple[int, ...]) -> bool:
    return len(shape) == 4 and key.endswith(".weight")


def _is_conv3d_weight(key: str, shape: tuple[int, ...]) -> bool:
    return len(shape) == 5 and key.endswith(".weight")


def convert_state_dict(sd: dict, strip_prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        nk = k
        if strip_prefix and nk.startswith(strip_prefix):
            nk = nk[len(strip_prefix):]
        arr = _to_numpy(v)
        shape = tuple(arr.shape)
        if _is_conv3d_weight(nk, shape):
            arr = _convert_conv3d_weight(arr)
        elif _is_conv2d_weight(nk, shape):
            arr = _convert_conv2d_weight(arr)
        out[nk] = arr
    return out


def convert_raft(pth_path: Path, out_path: Path) -> None:
    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    npz = convert_state_dict(sd, strip_prefix="module.")
    np.savez(str(out_path), **npz)
    print(f"  RAFT: {len(npz)} tensors  ({sum(v.size for v in npz.values()):,} params)")


def convert_rfc(pth_path: Path, out_path: Path) -> None:
    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    npz = convert_state_dict(sd)
    np.savez(str(out_path), **npz)
    print(f"  RFC:  {len(npz)} tensors  ({sum(v.size for v in npz.values()):,} params)")


def convert_main(pth_path: Path, out_path: Path) -> None:
    sd = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    npz = convert_state_dict(sd)
    np.savez(str(out_path), **npz)
    print(f"  MAIN: {len(npz)} tensors  ({sum(v.size for v in npz.values()):,} params)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default="weights/propainter-pt",
                        help="Directory with the 3 .pth files.")
    parser.add_argument("--out-dir", default="weights/propainter-mlx",
                        help="Output directory for the .npz files.")
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Converting from {src} -> {out}")
    convert_raft(src / "raft-things.pth", out / "raft.npz")
    convert_rfc(src / "recurrent_flow_completion.pth", out / "rfc.npz")
    convert_main(src / "ProPainter.pth", out / "propainter_main.npz")
    print("Done.")


if __name__ == "__main__":
    main()
