"""Convert a 3DGS .ply (INRIA / our infer_video format) to .spz (Niantic).

SPZ format dramatically compresses 3DGS scenes (typically 8-12× smaller
than .ply) for fast web loading. Compatible with SuperSplat, our local
viewer, and Niantic's reference player.

Usage:
    python3 meadow_sb/scripts/ply_to_spz.py <in.ply> [out.spz]

Workaround for a known import bug in the `spz` pip package: it ships an
__init__.py that does `from spz import ...` (which recurses on itself),
shadowing the actual native .so. We bypass by prepending the package
dir to sys.path so the .so is imported directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# spz pip package has a circular import — load the C-extension directly.
_SPZ_DIR = "/opt/homebrew/lib/python3.11/site-packages/spz"
if _SPZ_DIR not in sys.path:
    sys.path.insert(0, _SPZ_DIR)
import spz  # noqa: E402


# ----- .ply (INRIA 3DGS) reader ------------------------------------------- #

def read_3dgs_ply(path: Path) -> dict:
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        body = f.read()

    # Parse header to find num vertices + property list.
    lines = header.decode("ascii", errors="ignore").splitlines()
    n_vertex = None
    props = []
    for ln in lines:
        if ln.startswith("element vertex"):
            n_vertex = int(ln.split()[-1])
        elif ln.startswith("property"):
            parts = ln.split()
            # property float NAME
            props.append((parts[-1], parts[1]))

    if n_vertex is None or not props:
        raise ValueError(f"unrecognised .ply header in {path}")

    # All standard 3DGS .ply use float32 per property.
    dtype = np.dtype([(name, _np_dtype(t)) for name, t in props])
    arr = np.frombuffer(body, dtype=dtype, count=n_vertex)
    return {name: arr[name].astype(np.float32) for name in arr.dtype.names}


def _np_dtype(ply_type: str) -> str:
    return {
        "float": "<f4", "float32": "<f4",
        "double": "<f8", "float64": "<f8",
        "uchar": "<u1", "uint8": "<u1",
    }.get(ply_type, "<f4")


# ----- 3DGS conventions ---------------------------------------------------- #
# (Matches INRIA 3DGS / our infer_video.py writer.)
#
#   stored xyz       = linear, in world coordinates
#   stored scale_X   = log(linear scale)        → exp() to recover
#   stored opacity   = logit(true opacity)      → sigmoid() to recover
#   stored f_dc_X    = SH DC raw coefficient
#                      RGB colour = sigmoid(0.282094791773878 * f_dc + 0.5)
#   stored rot_X     = wxyz unit quaternion
# --------------------------------------------------------------------------- #

SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ply_dict_to_spz_arrays(d: dict) -> dict:
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=-1)
    log_scales = np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], axis=-1)
    scales = np.exp(log_scales)  # linear, metres-ish
    # quaternion stored as wxyz — SPZ also wants wxyz (matches PlayCanvas conv)
    rot = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], axis=-1)
    rot = rot / (np.linalg.norm(rot, axis=-1, keepdims=True) + 1e-12)
    alphas = sigmoid(d["opacity"])
    # SH DC → linear-space RGB in [0, 1]
    f_dc = np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], axis=-1)
    colors = sigmoid(SH_C0 * f_dc + 0.5).clip(0.0, 1.0)
    return {
        "positions": xyz.astype(np.float32),
        "scales": scales.astype(np.float32),
        "rotations": rot.astype(np.float32),
        "alphas": alphas.astype(np.float32),
        "colors": colors.astype(np.float32),
    }


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 ply_to_spz.py <in.ply> [out.spz]")
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_path.with_suffix(".spz")

    print(f"[ply_to_spz] reading {in_path} ({in_path.stat().st_size/1e6:.1f} MB) …")
    d = read_3dgs_ply(in_path)
    n = d["x"].shape[0]
    print(f"   {n:,} Gaussians, {len(d)} properties per vertex")

    arrs = ply_dict_to_spz_arrays(d)
    print(f"   xyz range: x[{arrs['positions'][:,0].min():.2f},{arrs['positions'][:,0].max():.2f}] "
          f"y[{arrs['positions'][:,1].min():.2f},{arrs['positions'][:,1].max():.2f}] "
          f"z[{arrs['positions'][:,2].min():.2f},{arrs['positions'][:,2].max():.2f}]")
    print(f"   alpha range: [{arrs['alphas'].min():.3f}, {arrs['alphas'].max():.3f}]")
    print(f"   color range: [{arrs['colors'].min():.3f}, {arrs['colors'].max():.3f}]")
    print(f"   scale range (linear m): [{arrs['scales'].min():.4f}, {arrs['scales'].max():.4f}]")

    g = spz.GaussianSplat(**arrs)
    g.save(str(out_path))
    print(f"[ply_to_spz] wrote {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB; "
          f"{in_path.stat().st_size / out_path.stat().st_size:.1f}× smaller)")


if __name__ == "__main__":
    main()
