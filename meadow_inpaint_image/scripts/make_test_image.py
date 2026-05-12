"""Synthesize a 512x512 RGB test image and a mask, save under tests/data/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    H = W = 512
    rng = np.random.default_rng(0)

    # gradient base
    ys = np.linspace(0, 1, H, dtype=np.float32)
    xs = np.linspace(0, 1, W, dtype=np.float32)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    R = (np.sin(XX * 6.0) * 0.5 + 0.5)
    G = (np.cos(YY * 5.0) * 0.5 + 0.5)
    B = (np.sin((XX + YY) * 7.0) * 0.5 + 0.5)
    img = np.stack([R, G, B], axis=-1)

    # add textured noise
    img = img * 0.7 + rng.uniform(0, 0.3, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img_u8 = (img * 255).astype(np.uint8)

    # shapes on top, so there is "content" to remove
    pim = Image.fromarray(img_u8)
    d = ImageDraw.Draw(pim)
    d.rectangle([80, 80, 220, 220], fill=(220, 30, 30))
    d.ellipse([260, 60, 460, 260], fill=(30, 200, 60))
    d.rectangle([100, 300, 400, 360], fill=(20, 30, 200))
    Image.fromarray(np.array(pim)).save(OUT / "image.png")

    # mask: white rectangle covering parts of the shapes
    msk = np.zeros((H, W), dtype=np.uint8)
    msk[120:200, 120:380] = 255  # horizontal band over red+green
    msk[330:400, 200:340] = 255  # over blue bar
    Image.fromarray(msk, mode="L").save(OUT / "mask.png")
    print(f"wrote {OUT/'image.png'} and {OUT/'mask.png'}")


if __name__ == "__main__":
    main()
