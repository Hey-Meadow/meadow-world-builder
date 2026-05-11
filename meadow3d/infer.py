"""CLI entry: SAM 3D Objects MLX inference.

Usage:
    python meadow3d/infer_mlx.py \
        --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
        --mask notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png \
        --seed 42 --out splat.ply

    # RGBA-merged input (mask in alpha channel):
    python meadow3d/infer_mlx.py --rgba combined.png --seed 42 --out splat.ply

    # Also emit a Niantic .spz alongside the .ply (~5x smaller):
    python meadow3d/infer_mlx.py --image X --mask Y --out splat.ply --format both
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make package importable when run directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from meadow3d.models.decoder_mlx import save_gaussian_ply  # noqa: E402
from meadow3d.models.pipeline_mlx import SAM3DObjectsPipeline  # noqa: E402


def load_rgba(image_path: str, mask_path: str | None = None) -> np.ndarray:
    """Load RGB image + binary mask (or RGBA combined) into an HxWx4 uint8 array."""
    img = Image.open(image_path)
    img = np.array(img.convert("RGBA")) if mask_path is not None else np.array(img)
    if mask_path is not None:
        mask_pil = Image.open(mask_path)
        mask = np.array(mask_pil)
        if mask.ndim == 3:
            mask = mask[..., -1]  # use alpha channel of mask file (matches PT load_mask)
        mask = (mask > 0).astype(np.uint8) * 255
        # Replace alpha channel with the supplied mask (matches PT merge_image_and_mask).
        rgba = np.concatenate([img[..., :3], mask[..., None]], axis=-1).astype(np.uint8)
        return rgba
    if img.ndim == 3 and img.shape[-1] == 4:
        return img.astype(np.uint8)
    raise ValueError(
        f"Without --mask, image must already be RGBA. Got shape {img.shape}."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=False, help="RGB or RGBA image path")
    ap.add_argument("--mask", required=False, help="Mask path (PNG); used as alpha")
    ap.add_argument("--rgba", required=False, help="Pre-merged RGBA image path")
    ap.add_argument(
        "--auto-mask", action="store_true",
        help="Generate mask via MLX SAM3 (community port). "
             "Combined with --image (any format) and optional --auto-mask-prompt.",
    )
    ap.add_argument(
        "--auto-mask-prompt", default="main object",
        help="Text prompt for SAM3 segmentation. Defaults to 'main object'; "
             "for best results name the subject (e.g. 'plush toy').",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="splat.ply")
    ap.add_argument(
        "--format", choices=["ply", "spz", "both"], default="ply",
        help="output format(s): ply only, spz only, or both. The .spz path "
             "is derived from --out by replacing the suffix.",
    )
    ap.add_argument("--ss-steps", type=int, default=25)
    ap.add_argument("--slat-steps", type=int, default=25)
    ap.add_argument("--ss-cfg", type=float, default=7.0)
    ap.add_argument("--slat-cfg", type=float, default=5.0)
    # ---- Shortcut model (SS only) -----------------------------------------
    # The SS DiT was self-distilled with the ShortCut objective
    # (arXiv:2410.12557): the SS .npz ships a `d_embedder` MLP that lets us
    # replace the 25-step CFG-7 sampler with a 4-step pass (no CFG) through
    # the same backbone. SLAT has no d_embedder so its sampler is untouched.
    ss_short_grp = ap.add_mutually_exclusive_group()
    ss_short_grp.add_argument(
        "--use-shortcut", action="store_true",
        help="SS stage: use distilled shortcut sampler (4 steps, no CFG).",
    )
    ss_short_grp.add_argument(
        "--no-shortcut", action="store_true",
        help="SS stage: force the 25-step CFG-7 baseline (default).",
    )
    ap.add_argument(
        "--shortcut-steps", type=int, default=4,
        help="SS shortcut step count (only used when --use-shortcut is set).",
    )
    ap.add_argument(
        "--npz-dir", default="meadow3d/weights/sam3d_objects",
        help="Directory containing the converted npz weights",
    )
    ap.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16", "mixed"], default="mixed",
        help="Precision policy for the SS / SLAT DiT backbones. "
             "'mixed' (default) = bf16 inside transformer blocks (matches PT "
             "torch.autocast(bfloat16) wrapper) with fp32 elsewhere. "
             "'bf16' is an alias for 'mixed'. 'fp16' uses float16 (matches PT "
             "default flash_attn precision). 'fp32' disables the cast.",
    )
    pm_grp = ap.add_mutually_exclusive_group()
    pm_grp.add_argument(
        "--use-moge", action="store_true",
        help="Use real MoGe pointmap (DINOv2 ViT-L, ~310M params). Default.",
    )
    pm_grp.add_argument(
        "--dummy-pointmap", action="store_true",
        help="Use synthetic constant-z dummy pointmap (degraded output).",
    )
    # Floating-outlier prune (post GS-decode). Default ON: drops voxels
    # whose voxel-grid neighborhood holds fewer than --outlier-min-neighbors
    # points within --outlier-radius (in grid units). Removes the visible
    # speckle of high-opacity Gaussians floating above/around the main mass.
    prune_grp = ap.add_mutually_exclusive_group()
    prune_grp.add_argument(
        "--prune-outliers", dest="prune_outliers", action="store_true",
        help="Drop isolated floating Gaussians via local voxel density (default).",
    )
    prune_grp.add_argument(
        "--no-prune-outliers", dest="prune_outliers", action="store_false",
        help="Disable the floating-outlier prune.",
    )
    ap.set_defaults(prune_outliers=True)
    ap.add_argument(
        "--outlier-radius", type=float, default=2.0,
        help="Voxel-grid radius for outlier neighborhood query.",
    )
    ap.add_argument(
        "--input-size", type=int, default=518,
        help="Side length (in pixels) fed to the DINO image trunk and "
             "PointPatchEmbed pointmap branch. Default 518 matches PT "
             "(canonical DINOv2 grid 37x37). Use 1024 for a 73x73 grid "
             "(~4x compute, finer detail; pos_embed is bilinearly "
             "interpolated). Must be a multiple of 14.",
    )
    ap.add_argument(
        "--outlier-min-neighbors", type=int, default=3,
        help="Voxels with fewer neighbors (incl. self) within radius are pruned.",
    )
    args = ap.parse_args()
    # Default = MoGe (real depth) unless --dummy-pointmap is given.
    use_moge = not args.dummy_pointmap

    if args.auto_mask:
        from meadow3d.utils.auto_mask import auto_mask_image
        if args.image is None and args.rgba is None:
            ap.error("--auto-mask requires --image or --rgba (raw image to segment)")
            return
        src_path = args.image if args.image is not None else args.rgba
        t_mask = time.time()
        rgba = auto_mask_image(src_path, text_prompt=args.auto_mask_prompt)
        print(f"[auto-mask] SAM3 done in {time.time()-t_mask:.1f}s "
              f"(prompt={args.auto_mask_prompt!r}, mask coverage="
              f"{100*(rgba[..., 3] > 0).mean():.1f}%)")
    elif args.rgba is not None:
        rgba = load_rgba(args.rgba)
    elif args.image is not None and args.mask is not None:
        rgba = load_rgba(args.image, args.mask)
    elif args.image is not None:
        rgba = load_rgba(args.image)
    else:
        ap.error("must supply either --rgba, --image (+ --mask or --auto-mask)")
        return

    print(f"Input shape: {rgba.shape}")

    t_load = time.time()
    pipeline = SAM3DObjectsPipeline.from_npz_dir(args.npz_dir, dtype=args.dtype)
    print(f"[load] from_npz_dir done in {time.time() - t_load:.1f} s "
          f"(dtype={args.dtype})")

    # Resolve shortcut decision: explicit --use-shortcut wins; explicit
    # --no-shortcut wins; else auto-enable if the SS backbone has a d_embedder
    # (i.e. the checkpoint was self-distilled).
    has_d_embedder = bool(getattr(pipeline.ss_backbone, "has_d_embedder", False))
    if args.use_shortcut:
        use_shortcut = True
    elif args.no_shortcut:
        use_shortcut = False
    else:
        use_shortcut = has_d_embedder
    print(
        f"[run] SS sampler: "
        f"{'shortcut (' + str(args.shortcut_steps) + '-step, no CFG)' if use_shortcut else f'baseline ({args.ss_steps}-step CFG-{args.ss_cfg})'}"
    )
    print(f"[run] pointmap source: {'MoGe (real depth)' if use_moge else 'dummy'}")
    print(f"[run] input_size={args.input_size} (DINO grid {args.input_size//14}x{args.input_size//14})")
    t_run = time.time()
    out = pipeline(
        rgba_uint8=rgba,
        seed=args.seed,
        ss_steps=args.ss_steps,
        slat_steps=args.slat_steps,
        ss_cfg=args.ss_cfg,
        slat_cfg=args.slat_cfg,
        use_moge=use_moge,
        prune_outliers=args.prune_outliers,
        outlier_radius=args.outlier_radius,
        outlier_min_neighbors=args.outlier_min_neighbors,
        use_shortcut=use_shortcut,
        shortcut_steps=args.shortcut_steps,
        input_size=args.input_size,
    )
    print(f"[run] full pipeline in {time.time() - t_run:.1f} s")
    print(f"[run] timing per stage:")
    for k, v in out["timing"].items():
        if isinstance(v, (int, float)):
            print(f"  {k:<18} {v:7.2f} s")
        else:
            print(f"  {k:<18} {v}")
    print(f"[run] {out['n_voxels']} voxels -> {out['n_voxels'] * 32} Gaussians")

    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # We always need the .ply on disk first - save_gaussian_ply applies the
    # canonical 3DGS activations, and ply_to_spz reuses that representation.
    # If the user only wants .spz, we delete the intermediate .ply at the end.
    ply_path = (
        out_path
        if out_path.endswith(".ply")
        else str(Path(out_path).with_suffix(".ply"))
    )
    save_gaussian_ply(out["gs_params"], ply_path)
    if args.format in ("ply", "both"):
        print(f"[save] {ply_path} ({os.path.getsize(ply_path) / 1e6:.2f} MB)")

    if args.format in ("spz", "both"):
        # Lazy import so a missing `spz` package doesn't break --format ply.
        from meadow3d.scripts.ply_to_spz import ply_to_spz  # noqa: E402
        spz_path = (
            out_path
            if out_path.endswith(".spz")
            else str(Path(out_path).with_suffix(".spz"))
        )
        stats = ply_to_spz(ply_path, spz_path)
        print(
            f"[save] {spz_path} ({stats['spz_bytes']/1e6:.2f} MB, "
            f"{stats['ratio']:.2f}x vs ply)"
        )
        if args.format == "spz" and ply_path != out_path:
            # User asked for spz only and we'd been told to write to a
            # non-.ply --out (so the .ply is purely intermediate). Clean it up.
            try:
                os.remove(ply_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
