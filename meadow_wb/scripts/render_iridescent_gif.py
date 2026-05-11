"""One-shot iridescent 3DGS turntable GIF generator.

For each frame i ∈ [0, n_frames):
  1. Rotate the input .ply around Y by 2π·i/n_frames
  2. Apply `apply_iridescent.py` with phase = i/n_frames
  3. Render via macOS Quick Look
  4. Gaussian-blur the PNG for a dreamy soft look
Stitch into an animated .gif.

Usage:
    python meadow_wb/scripts/render_iridescent_gif.py \\
        assets/demos/chair_clean.ply  out.gif \\
        --frames 36 --metallic-mix 0.95 --blur 2.5 --size 320
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _read_ply_struct(path):
    with open(path, "rb") as f:
        hdr = []
        while True:
            ln = f.readline()
            hdr.append(ln)
            if ln.strip() == b"end_header":
                break
        header = b"".join(hdr)
        h = header.decode()
        n = 0
        props = []
        for ln in h.splitlines():
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            if ln.startswith("property"):
                props.append((ln.split()[1], ln.split()[2]))
        dt = {"float": "f4", "double": "f8"}
        nd = np.dtype([(name, dt[t]) for t, name in props])
        body = np.fromfile(f, dtype=nd, count=n)
    return header, body


def _rotate_y(header, body, angle_rad, out_path):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    xyz = np.stack([body["x"], body["y"], body["z"]], axis=1).astype(np.float32) @ R.T
    out = body.copy()
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    with open(out_path, "wb") as f:
        f.write(header)
        out.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out_gif")
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--freq", type=float, default=4.0)
    ap.add_argument("--metallic-mix", type=float, default=0.95)
    ap.add_argument("--blur", type=float, default=2.5,
                    help="Gaussian blur radius applied at 1.5× render size before downscale")
    ap.add_argument("--duration-ms", type=int, default=80)
    args = ap.parse_args()

    here = Path(__file__).parent.resolve()
    iri_script = here / "apply_iridescent.py"
    repo_python = sys.executable

    work = Path(tempfile.mkdtemp(prefix="iri_gif_"))
    try:
        header, body = _read_ply_struct(args.inp)
        big = args.size * 3 // 2  # 1.5× render then downsample for soft AA
        frames = []
        for i in range(args.frames):
            ang = 2 * math.pi * i / args.frames
            phase = i / args.frames
            rot_ply = work / f"rot_{i:03d}.ply"
            iri_ply = work / f"iri_{i:03d}.ply"
            _rotate_y(header, body, ang, rot_ply)
            subprocess.run(
                [
                    repo_python, str(iri_script),
                    str(rot_ply), str(iri_ply),
                    "--view", "0", "0", "1",
                    "--freq", str(args.freq),
                    "--phase", str(phase),
                    "--normal", "radial",
                    "--metallic-mix", str(args.metallic_mix),
                ],
                check=True, capture_output=True,
            )
            # Render via macOS Quick Look
            subprocess.run(
                ["qlmanage", "-x", "-t", "-s", str(big), "-o", str(work), str(iri_ply)],
                check=False, capture_output=True,
            )
            png = work / f"{iri_ply.name}.png"
            im = Image.open(png).convert("RGB")
            if args.blur > 0:
                im = im.filter(ImageFilter.GaussianBlur(radius=args.blur))
            im = im.resize((args.size, args.size * im.height // im.width), Image.LANCZOS)
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=96))

        frames[0].save(
            args.out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=args.duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(f"[iri-gif] {args.out_gif}  {args.frames} frames  {Path(args.out_gif).stat().st_size/1024:.0f} KB")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
