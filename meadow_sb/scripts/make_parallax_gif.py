"""Render a small orbit around view-0 to show 3D parallax in the .ply.

Loads the YoNoSplat MLX assembler output (or a saved .npz), then renders
N frames with the camera doing a tiny side-to-side sweep about view-0,
and saves them as an animated GIF (+ optional MP4).

Usage:
    python3.11 meadow_sb/scripts/make_parallax_gif.py \
        --video <path/to/video.mp4> \
        --out  meadow_sb/media/scene.gif \
        --gt   meadow_sb/media/scene_gt.png \
        --render meadow_sb/media/scene_render.png \
        --n-frames 8 --orbit-frames 12 --deg 4
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import imageio.v2 as imageio

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from meadow_sb.models.yonosplat import YoNoSplatEncoder, YoNoSplatEncoderCfg  # noqa: E402
from meadow_sb.models.rasterizer import Gaussians as RG, GsplatRasterizer  # noqa: E402
from meadow_sb.scripts.infer_video import sample_n_frames  # noqa: E402

CKPT = Path(__file__).resolve().parents[2] / "research" / "yonosplat_bootstrap" / \
       "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"


def _rotate_about_y_through_point(p: np.ndarray, axis_origin: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 3D point p around the world Y axis (passing through axis_origin) by `deg`."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return R @ (p - axis_origin) + axis_origin


def _orbit_c2w(view0_c2w: np.ndarray, scene_center: np.ndarray, deg: float) -> np.ndarray:
    """Rotate camera position around scene_center by `deg` about world Y, keep look-at scene_center."""
    eye = view0_c2w[:3, 3].astype(np.float32)
    eye_new = _rotate_about_y_through_point(eye, scene_center, deg)
    # camera-forward = (scene_center - eye_new), normalised
    fwd = scene_center - eye_new
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    # original up = -world Y (OpenCV: +Y is image-down → world up = -Y)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_corr = np.cross(right, fwd)
    R = np.stack([right, -up_corr, fwd], axis=1)   # camera basis: X=right, Y=-up (OpenCV), Z=fwd
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye_new
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output GIF path")
    ap.add_argument("--gt", type=Path, default=None, help="also save the GT view-0 PNG")
    ap.add_argument("--render", type=Path, default=None, help="also save the static view-0 render PNG")
    ap.add_argument("--n-frames", type=int, default=8, help="N views into the MLX forward")
    ap.add_argument("--orbit-frames", type=int, default=12, help="frames in the parallax loop")
    ap.add_argument("--deg", type=float, default=4.0, help="orbit amplitude (degrees, ±)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--size", type=int, default=224, help="render resolution")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[parallax] loading ckpt …")
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)["state_dict"]
    print(f"[parallax] building model …")
    model = YoNoSplatEncoder(YoNoSplatEncoderCfg(), state_dict=sd)

    print(f"[parallax] sampling {args.n_frames} frames from {args.video} …")
    frames = sample_n_frames(args.video, args.n_frames, size=args.size)
    images_mx = mx.array(frames[None, ...].astype(np.float32))
    K_norm = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float32)
    intr_mx = mx.broadcast_to(mx.array(K_norm).reshape(1, 1, 3, 3), (1, args.n_frames, 3, 3))

    t0 = time.time()
    out = model(images_mx, intr_mx)
    g = out["gaussians"]
    mx.eval(g.means, g.scales, g.rotations, g.opacities, g.harmonics,
            out["camera_poses"], out["intrinsic_pred"])
    print(f"[parallax] forward in {time.time()-t0:.2f}s")

    # Pull tensors for rasterizer (PT side, OpenCV frame)
    means = np.asarray(g.means).reshape(1, -1, 3)
    scales = np.asarray(g.scales).reshape(1, -1, 3)
    rots_xyzw = np.asarray(g.rotations).reshape(1, -1, 4)
    rots_wxyz = rots_xyzw[..., [3, 0, 1, 2]]
    ops = np.asarray(g.opacities).reshape(1, -1, 1)
    sh = np.asarray(g.harmonics).reshape(1, -1, 3, 1)
    gp = RG(xyz=torch.from_numpy(means).float(),
            scale=torch.from_numpy(scales).float(),
            rotation=torch.from_numpy(rots_wxyz).float(),
            opacity=torch.from_numpy(ops).float(),
            features=torch.from_numpy(sh).float())

    intr_pred = np.asarray(out["intrinsic_pred"])[0]                          # (V, 2)
    K = np.zeros((1, 1, 3, 3), dtype=np.float32)
    K[..., 0, 0] = intr_pred[0, 0]
    K[..., 1, 1] = intr_pred[0, 1]
    K[..., 0, 2] = 0.5
    K[..., 1, 2] = 0.5
    K[..., 2, 2] = 1.0

    # Scene "centre" for the orbit. Use the median of the local_points cluster.
    lp = np.asarray(out["local_points"]).reshape(-1, 3)
    scene_center = np.median(lp[(lp[:, 2] > 0.1) & (lp[:, 2] < 50)], axis=0).astype(np.float32)
    print(f"[parallax] orbit center = {scene_center}")

    rasterizer = GsplatRasterizer(opacity_threshold=0.005)

    # Optional: static view-0 render + GT save
    cam0 = np.asarray(out["camera_poses"])[0, 0]                              # (4, 4)
    if args.render is not None:
        c2w_t = torch.from_numpy(cam0[None, None]).float()
        rgb = rasterizer.render(gp, c2w_t, torch.from_numpy(K), (args.size, args.size))
        img = (rgb[0, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(str(args.render), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[parallax] static render → {args.render}")
    if args.gt is not None:
        gt = (frames[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(str(args.gt), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
        print(f"[parallax] GT view-0 → {args.gt}")

    # Orbit
    frames_gif = []
    print(f"[parallax] rendering {args.orbit_frames} orbit frames (±{args.deg}°) …")
    for i in range(args.orbit_frames):
        # phase: 0 → +deg → 0 → -deg → 0 (smooth ping-pong)
        phase = np.sin(2 * np.pi * i / args.orbit_frames) * args.deg
        c2w = _orbit_c2w(cam0, scene_center, phase)
        c2w_t = torch.from_numpy(c2w[None, None]).float()
        t0 = time.time()
        rgb = rasterizer.render(gp, c2w_t, torch.from_numpy(K), (args.size, args.size))
        img = (rgb[0, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        frames_gif.append(img)
        print(f"   frame {i+1}/{args.orbit_frames}  phase={phase:+.2f}°  ({time.time()-t0:.1f}s)")

    imageio.mimsave(str(args.out), frames_gif, fps=args.fps, loop=0)
    size_kb = args.out.stat().st_size / 1024
    print(f"[parallax] wrote {args.out} ({size_kb:.0f} KB, {args.orbit_frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
