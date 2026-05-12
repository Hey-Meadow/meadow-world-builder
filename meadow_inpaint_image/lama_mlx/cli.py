"""Console entry point for `lama-mlx-inpaint`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .generator import FFCResNetGenerator
from .inference import inpaint


def main():
    ap = argparse.ArgumentParser(prog="lama-mlx-inpaint", description="MLX LaMa inpainting CLI")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True, help="Path to lama_mlx.npz")
    args = ap.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    msk = np.array(Image.open(args.mask).convert("L"))
    print(f"image {img.shape}, mask {msk.shape}")
    t0 = time.time()
    model = FFCResNetGenerator.from_npz(args.weights)
    print(f"loaded weights in {time.time()-t0:.2f}s")
    t0 = time.time()
    out = inpaint(model, img, msk)
    print(f"inpainted in {time.time()-t0:.2f}s")
    Image.fromarray(out).save(args.out)
    print(f"wrote {args.out}")
