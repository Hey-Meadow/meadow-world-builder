"""Video → 3DGS .ply via MLX YoNoSplat.

Reads a video, samples two well-separated frames as the YoNoSplat 2-view
context, runs the MLX assembler, and writes a standard 3DGS `.ply` that
opens in SuperSplat / Polycam / antimatter15's viewer.

Usage:
    python3.11 meadow_sb/scripts/infer_video.py \
        --video path/to/clip.mp4 \
        --out scene.ply
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from meadow_sb.models.yonosplat import (  # noqa: E402
    YoNoSplatEncoder,
    YoNoSplatEncoderCfg,
)

CKPT = Path(__file__).resolve().parents[2] / "research" / "yonosplat_bootstrap" / \
       "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"


# --------------------------------------------------------------------------- #
# Video frame sampling
# --------------------------------------------------------------------------- #

def sample_n_frames(video_path: Path, n: int, size: int = 224) -> np.ndarray:
    """Read `n` frames uniformly spaced across the clip.

    Returns:
        np.ndarray of shape (n, 3, size, size), float32 in [0, 1].
    """
    if n < 2:
        raise ValueError(f"need at least 2 frames, got n={n}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cv2 could not open {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Try seek-based fast path first; for many screen recordings the
    # container reports an inflated nb_frames vs what the decoder can
    # actually deliver, so fall back to a full sequential scan if seek
    # fails. Scanning is the safe path (variable-FPS aware).
    used_scan = False
    if n_frames > 1:
        idxs = [int(round(t * (n_frames - 1))) for t in np.linspace(0, 1, n)]
        chosen = []
        ok_all = True
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                ok_all = False
                break
            chosen.append(fr)
        if not ok_all:
            chosen = []
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
            used_scan = True
    else:
        used_scan = True

    if used_scan:
        frames_all = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames_all.append(fr)
        cap.release()
        n_actual = len(frames_all)
        if n_actual < n:
            raise ValueError(f"video has {n_actual} frame(s), need {n}")
        idxs = [int(round(t * (n_actual - 1))) for t in np.linspace(0, 1, n)]
        chosen = [frames_all[i] for i in idxs]
        n_frames = n_actual
    else:
        cap.release()

    out = np.zeros((n, 3, size, size), dtype=np.float32)
    for i, fr in enumerate(chosen):
        h, w = fr.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        fr = fr[y0:y0 + s, x0:x0 + s]
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        out[i] = fr_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    print(f"[infer-video] sampled {n} frames from indices {idxs} / {n_frames}")
    return out


# Back-compat alias for any older caller.
def sample_two_frames(video_path: Path, size: int = 224, frame_a: float = 0.0,
                      frame_b: float = 0.5) -> np.ndarray:
    return sample_n_frames(video_path, 2, size=size)


# --------------------------------------------------------------------------- #
# .ply writer (standard 3DGS schema — opens in SuperSplat)
# --------------------------------------------------------------------------- #

def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Adapter emits xyzw quaternions; .ply convention is wxyz."""
    return q[..., [3, 0, 1, 2]]


def _flip_yz_xyz(xyz: np.ndarray) -> np.ndarray:
    """OpenCV (+Y down, +Z forward) → OpenGL/SuperSplat (+Y up, -Z forward).

    Composed as a 180° rotation around the X axis: diag(1, -1, -1).
    """
    out = xyz.copy()
    out[:, 1] = -xyz[:, 1]
    out[:, 2] = -xyz[:, 2]
    return out


def _flip_yz_quat(rot_xyzw: np.ndarray) -> np.ndarray:
    """Same diag(1, -1, -1) world transform applied to per-Gaussian rotation.

    Pre-multiply by q_M = (x=1, y=0, z=0, w=0) (= 180° around X-axis):
      q_new (xyzw) = (w_old, -z_old, y_old, -x_old)
    """
    x, y, z, w = rot_xyzw[:, 0], rot_xyzw[:, 1], rot_xyzw[:, 2], rot_xyzw[:, 3]
    return np.stack([w, -z, y, -x], axis=-1)


def _transform_camera_pose(c2w: np.ndarray) -> np.ndarray:
    """Apply the same world flip to camera-to-world extrinsics.

    M @ c2w, with M = diag(1, -1, -1, 1).
    """
    M = np.diag([1.0, -1.0, -1.0, 1.0]).astype(c2w.dtype)
    return M @ c2w


def _log_scale(s: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """3DGS .ply stores log-scale (linear scale is exp() of stored value)."""
    return np.log(np.clip(s, eps, None))


def _logit_opacity(o: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """3DGS .ply stores logit-opacity (true opacity is sigmoid of stored)."""
    o = np.clip(o, eps, 1.0 - eps)
    return np.log(o / (1.0 - o))


def write_ply_3dgs(path: Path, xyz: np.ndarray, scales: np.ndarray,
                   rotations_xyzw: np.ndarray, opacities: np.ndarray,
                   colours_dc: np.ndarray) -> None:
    """Write a standard 3DGS .ply.

    All inputs are (N, ...) numpy arrays:
      xyz             : (N, 3)
      scales          : (N, 3) linear (we log inside)
      rotations_xyzw  : (N, 4) unit quaternion in xyzw
      opacities       : (N,)   in [0, 1] (we logit inside)
      colours_dc      : (N, 3) SH DC term (raw, as stored upstream)
    """
    N = xyz.shape[0]
    assert scales.shape == (N, 3)
    assert rotations_xyzw.shape == (N, 4)
    assert opacities.shape == (N,)
    assert colours_dc.shape == (N, 3)

    quat_wxyz = _quat_xyzw_to_wxyz(rotations_xyzw)
    log_scales = _log_scale(scales)
    logit_op = _logit_opacity(opacities)

    # Build the structured record. Order matches SuperSplat / antimatter15 reader.
    props = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),   # unused but expected
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    rec = np.zeros(N, dtype=props)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # leave normals zero
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = colours_dc[:, 0], colours_dc[:, 1], colours_dc[:, 2]
    rec["opacity"] = logit_op
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = (
        quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    )

    # Header
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    for name, dtype in props:
        header += f"property float {name}\n"
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    print(f"[infer-video] wrote {N:,} Gaussians to {path} ({path.stat().st_size/1e6:.1f} MB)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("scene.ply"))
    ap.add_argument("--n-frames", type=int, default=2,
                    help="number of frames to sample uniformly from the video "
                         "(YoNoSplat re10k ckpt trained for 2-32)")
    ap.add_argument("--frame-a", type=float, default=0.0, help="(deprecated, kept for back-compat)")
    ap.add_argument("--frame-b", type=float, default=0.5, help="(deprecated, kept for back-compat)")
    ap.add_argument("--opacity-threshold", type=float, default=0.005,
                    help="prune Gaussians below this opacity before writing .ply")
    ap.add_argument("--render-preview", action="store_true",
                    help="also render view-0 via the CPU rasterizer and save as PNG")
    ap.add_argument("--no-axis-flip", action="store_true",
                    help="keep raw OpenCV (+Y down, +Z fwd) — by default we flip "
                         "to OpenGL/SuperSplat (+Y up, -Z fwd)")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")

    print(f"[infer-video] loading ckpt …")
    t0 = time.time()
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)["state_dict"]
    print(f"   ckpt loaded ({len(sd)} tensors, {time.time()-t0:.1f}s)")

    print(f"[infer-video] building MLX YoNoSplat …")
    t0 = time.time()
    model = YoNoSplatEncoder(YoNoSplatEncoderCfg(), state_dict=sd)
    print(f"   model built ({time.time()-t0:.1f}s)")

    V = args.n_frames
    print(f"[infer-video] sampling {V} frames from {args.video} …")
    frames = sample_n_frames(args.video, V, size=224)
    images = frames[None, ...]                                  # (1, V, 3, 224, 224)
    images_mx = mx.array(images)
    K_norm = np.array(
        [[1.0, 0.0, 0.5],
         [0.0, 1.0, 0.5],
         [0.0, 0.0, 1.0]], dtype=np.float32
    )
    intr_mx = mx.broadcast_to(mx.array(K_norm).reshape(1, 1, 3, 3), (1, V, 3, 3))

    print(f"[infer-video] running MLX forward …")
    t0 = time.time()
    out = model(images_mx, intr_mx)
    # Force materialisation
    mx.eval(
        out["gaussians"].means, out["gaussians"].scales,
        out["gaussians"].rotations, out["gaussians"].opacities,
        out["gaussians"].harmonics,
    )
    print(f"   forward in {time.time()-t0:.2f}s")

    g = out["gaussians"]
    # (1, V, N_per_view, S=1, 1, 3) → (N_total, 3)
    means = np.asarray(g.means).reshape(-1, 3)
    scales = np.asarray(g.scales).reshape(-1, 3)
    rots = np.asarray(g.rotations).reshape(-1, 4)
    ops = np.asarray(g.opacities).reshape(-1)
    # harmonics shape (1, V, N, S, 1, 3, d_sh=1) → DC only
    sh = np.asarray(g.harmonics).reshape(-1, 3)

    print(f"[infer-video] {means.shape[0]:,} raw Gaussians, opacity range [{ops.min():.3f}, {ops.max():.3f}]")
    keep = ops > args.opacity_threshold
    print(f"   keeping {keep.sum():,} after opacity > {args.opacity_threshold}")

    xyz_w = means[keep]
    rot_w = rots[keep]
    cam_poses = np.asarray(out["camera_poses"])[0]  # (V, 4, 4) c2w

    if not args.no_axis_flip:
        print(f"[infer-video] axis-flip OpenCV → OpenGL/SuperSplat (+Y up, −Z fwd)")
        xyz_w = _flip_yz_xyz(xyz_w)
        rot_w = _flip_yz_quat(rot_w)
        cam_poses = np.stack([_transform_camera_pose(p) for p in cam_poses], axis=0)

    write_ply_3dgs(args.out, xyz_w, scales[keep], rot_w, ops[keep], sh[keep])

    # Sidecar JSON: per-view camera + the predicted intrinsics so a future
    # web viewer can spawn its initial camera at view-0 (= GT photo angle).
    import json
    sidecar = args.out.with_suffix(".cameras.json")
    intr_pred = np.asarray(out["intrinsic_pred"])[0]   # (V, 2)
    cams = []
    for v in range(cam_poses.shape[0]):
        cams.append({
            "view": int(v),
            "c2w": cam_poses[v].tolist(),
            "intrinsics_normalised": {
                "fx": float(intr_pred[v, 0]), "fy": float(intr_pred[v, 1]),
                "cx": 0.5, "cy": 0.5,
            },
        })
    with open(sidecar, "w") as f:
        json.dump({
            "axis_convention": "openCV" if args.no_axis_flip else "openGL_Yup",
            "image_size": [224, 224],
            "cameras": cams,
            "initial_view": 0,
        }, f, indent=2)
    print(f"[infer-video] camera sidecar → {sidecar}")

    if args.render_preview:
        render_preview(out, args, frames)

    print(f"[infer-video] done. Open {args.out} in SuperSplat to view.")


def render_preview(model_out: dict, args, frames_np: np.ndarray) -> None:
    """Render view-0 with the CPU rasterizer and save preview PNG next to .ply."""
    from meadow_sb.models.rasterizer import Gaussians as RGaussians, GsplatRasterizer

    g = model_out["gaussians"]
    means = np.asarray(g.means).reshape(1, -1, 3)
    scales = np.asarray(g.scales).reshape(1, -1, 3)
    rots_xyzw = np.asarray(g.rotations).reshape(1, -1, 4)
    rots_wxyz = rots_xyzw[..., [3, 0, 1, 2]]                    # rasterizer expects wxyz
    ops = np.asarray(g.opacities).reshape(1, -1, 1)
    sh_dc = np.asarray(g.harmonics).reshape(1, -1, 3, 1)         # (B, N, 3, d_sh=1)

    gp = RGaussians(
        xyz=torch.from_numpy(means).float(),
        scale=torch.from_numpy(scales).float(),
        rotation=torch.from_numpy(rots_wxyz).float(),
        opacity=torch.from_numpy(ops).float(),
        features=torch.from_numpy(sh_dc).float(),
    )

    c2w = torch.from_numpy(np.asarray(model_out["camera_poses"])).float()   # (1, V, 4, 4)
    # Build intrinsics from predicted (fx, fy), principal point at image centre.
    intr_pred = np.asarray(model_out["intrinsic_pred"])                      # (1, V, 2)
    K = np.zeros((1, intr_pred.shape[1], 3, 3), dtype=np.float32)
    K[..., 0, 0] = intr_pred[..., 0]
    K[..., 1, 1] = intr_pred[..., 1]
    K[..., 0, 2] = 0.5
    K[..., 1, 2] = 0.5
    K[..., 2, 2] = 1.0
    K_t = torch.from_numpy(K)

    rasterizer = GsplatRasterizer(opacity_threshold=args.opacity_threshold)
    t0 = time.time()
    print(f"[infer-video] rendering view-0 via CPU rasterizer …")
    rgb = rasterizer.render(gp, c2w[:, :1], K_t[:, :1], (224, 224))           # (1, 1, 3, H, W)
    print(f"   render in {time.time()-t0:.2f}s")
    rgb_np = rgb[0, 0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    rgb_u8 = (rgb_np * 255).astype(np.uint8)

    out_png = args.out.with_suffix(".rendered.png")
    cv2.imwrite(str(out_png), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    print(f"[infer-video] preview saved to {out_png}")

    # Also dump the GT frame for side-by-side comparison
    gt_png = args.out.with_suffix(".gt_view0.png")
    gt_u8 = (frames_np[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(gt_png), cv2.cvtColor(gt_u8, cv2.COLOR_RGB2BGR))
    print(f"[infer-video] GT view-0 saved to {gt_png}")


if __name__ == "__main__":
    main()
