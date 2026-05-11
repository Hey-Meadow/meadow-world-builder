"""Stitch a folder of PNGs (sorted by name) into an optimized GIF."""
from __future__ import annotations
import sys
from pathlib import Path
from PIL import Image


def main(png_dir: str, out_gif: str, duration_ms: int = 80,
         palette_colors: int = 96):
    pngs = sorted(Path(png_dir).glob("*.png"))
    if not pngs:
        raise SystemExit(f"no pngs in {png_dir}")
    frames = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        frames.append(im.convert("P", palette=Image.ADAPTIVE,
                                 colors=palette_colors))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True, disposal=2)
    print(f"  {len(frames)} frames -> {out_gif}")


if __name__ == "__main__":
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    main(sys.argv[1], sys.argv[2], duration_ms=duration)
