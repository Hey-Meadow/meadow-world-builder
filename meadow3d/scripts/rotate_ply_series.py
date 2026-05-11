"""Generate N rotated copies of a 3DGS .ply (spin around Y axis).

Loads the input ply once, rotates xyz in memory, writes N output plys.
~30× faster than calling rotate_ply.py 36 times.
"""
from __future__ import annotations
import sys, argparse, math
from pathlib import Path
import numpy as np


def main(in_path: str, out_dir: str, n_frames: int, axis: str = "y"):
    with open(in_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines)
        h = header.decode()
        n = 0
        props = []
        for ln in h.splitlines():
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            if ln.startswith("property"):
                props.append((ln.split()[1], ln.split()[2]))
        dm = {"float": "f4", "double": "f8"}
        nd = np.dtype([(name, dm[t]) for t, name in props])
        d = np.fromfile(f, dtype=nd, count=n)

    xyz = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32)
    stem = Path(in_path).stem
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        rad = 2 * math.pi * i / n_frames
        c, s = math.cos(rad), math.sin(rad)
        if axis == "y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        elif axis == "x":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        rotated = xyz @ R.T
        out = d.copy()
        out["x"] = rotated[:, 0]
        out["y"] = rotated[:, 1]
        out["z"] = rotated[:, 2]
        out_path = f"{out_dir}/{stem}_{i:03d}.ply"
        with open(out_path, "wb") as f:
            f.write(header)
            out.tofile(f)
    print(f"  {n_frames} plys -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--axis", default="y", choices=["x", "y", "z"])
    args = ap.parse_args()
    main(args.inp, args.out_dir, args.n, args.axis)
