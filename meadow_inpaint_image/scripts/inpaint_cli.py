"""Inpaint a single image+mask with the MLX LaMa generator.

Example:
  python3.11 scripts/inpaint_cli.py --image foo.png --mask m.png --out out.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lama_mlx.generator import FFCResNetGenerator
from lama_mlx.inference import inpaint


def load_image(p):
    return np.array(Image.open(p).convert("RGB"))


def load_mask(p):
    img = Image.open(p)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(ROOT / "weights" / "lama_mlx.npz"))
    args = ap.parse_args()

    print(f"Loading weights from {args.weights}")
    t0 = time.time()
    model = FFCResNetGenerator.from_npz(args.weights)
    print(f"  loaded in {time.time()-t0:.2f}s")

    img = load_image(args.image)
    msk = load_mask(args.mask)
    print(f"image {img.shape}, mask {msk.shape}, mask>127 ratio={(msk>127).mean():.3f}")
    assert img.shape[:2] == msk.shape[:2], "image/mask spatial dims mismatch"

    t0 = time.time()
    out = inpaint(model, img, msk)
    dt = time.time() - t0
    print(f"inpaint done in {dt:.2f}s  ({img.shape[0]}x{img.shape[1]})")

    Image.fromarray(out).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
