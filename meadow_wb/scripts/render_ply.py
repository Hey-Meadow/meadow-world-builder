"""render_ply.py — quick CPU preview of a 3DGS .ply.

Usage
-----
    python render_ply.py path/to/splat.ply
    python render_ply.py path/to/splat.ply --out preview --frames 60 --size 512
    python render_ply.py --synthetic                  # generate + render a random splat

Outputs (next to the .ply, or in --out dir):
    <stem>_preview.png        front view
    <stem>_turntable.mp4      360 deg spin (60 frames @ 30 fps)

Approach (Option B from SPEC_VIEWER.md)
---------------------------------------
Pure-numpy / OpenCV anisotropic Gaussian splatter.
Reads the PLY format written by ``meadow_wb/models/decoder_mlx.save_gaussian_ply``
(17 float32 props/vertex: xyz, nx ny nz, f_dc_0..2, opacity, scale_0..2,
rot_0..3) but also tolerates extra ``f_rest_*`` fields if a future writer
adds them.

For each Gaussian we
    1. Apply the same activations the official 3DGS viewer does
       (sigmoid on opacity, exp on log-scales, normalize quat, SH dc -> RGB).
    2. Build a 3x3 world-space covariance  Σ = R diag(s)^2 R^T.
    3. Project the centre with a perspective camera, project Σ via the
       2x3 Jacobian of the perspective map -> 2x2 image-space covariance.
    4. Front-to-back alpha-composite an oriented gaussian footprint
       within a 3-sigma bbox.

This is not a CUDA-grade rasterizer — it's deliberately simple so the
file stays short and dependency-light. It's enough to answer
"can I see object shape and colour".
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# PLY reading
# ---------------------------------------------------------------------------

def _read_ply_header(fp) -> Tuple[int, list, str]:
    """Return (vertex_count, [property_names], format_str). Pointer left at
    first byte of body."""
    line = fp.readline()
    if line.strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    n_vertex = 0
    props: list[str] = []
    in_vertex = False
    while True:
        line = fp.readline()
        if not line:
            raise ValueError("unexpected EOF in PLY header")
        s = line.decode("ascii", errors="replace").strip()
        if s.startswith("format"):
            fmt = s.split()[1]  # e.g. binary_little_endian
        elif s.startswith("element"):
            parts = s.split()
            in_vertex = (parts[1] == "vertex")
            if in_vertex:
                n_vertex = int(parts[2])
        elif s.startswith("property") and in_vertex:
            parts = s.split()
            # property <type> <name>
            if parts[1] != "float" and parts[1] != "float32":
                raise ValueError(
                    f"only float32 properties supported, got: {s!r}")
            props.append(parts[-1])
        elif s == "end_header":
            break
    if fmt is None:
        raise ValueError("PLY header missing format line")
    return n_vertex, props, fmt


def load_3dgs_ply(path: str) -> dict:
    """Read a 3DGS .ply written by save_gaussian_ply.

    Returns a dict of numpy arrays in the canonical (already-activated)
    form expected by a viewer:
        xyz       (N, 3)   world-space centres
        rgb       (N, 3)   in [0, 1]
        opacity   (N,)     in [0, 1]
        scale     (N, 3)   world-space sigmas
        quat      (N, 4)   unit (w, x, y, z)
    """
    with open(path, "rb") as fp:
        n, props, fmt = _read_ply_header(fp)
        body = fp.read()
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"only binary PLY supported here, got {fmt!r}")
    dtype = "<f4" if fmt == "binary_little_endian" else ">f4"
    arr = np.frombuffer(body, dtype=dtype, count=n * len(props))
    if arr.size != n * len(props):
        raise ValueError(
            f"body size {arr.size} != n*{len(props)} = {n * len(props)}")
    arr = arr.reshape(n, len(props)).astype(np.float32)

    def col(name: str) -> np.ndarray:
        return arr[:, props.index(name)]

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1)
    f_dc = np.stack(
        [col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    opacity_logit = col("opacity")
    log_scale = np.stack(
        [col("scale_0"), col("scale_1"), col("scale_2")], axis=1)
    quat = np.stack(
        [col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)

    # Activations matching gaussian_model.save_ply / 3DGS viewers.
    # f_dc holds SH degree-0 coefficients; sh0_to_rgb = 0.5 + C0 * sh0
    SH_C0 = 0.28209479177387814
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))           # sigmoid
    scale = np.exp(log_scale)                                # log -> sigma
    qn = np.linalg.norm(quat, axis=1, keepdims=True)
    qn = np.where(qn < 1e-8, 1.0, qn)
    quat = quat / qn

    return dict(xyz=xyz.astype(np.float32),
                rgb=rgb.astype(np.float32),
                opacity=opacity.astype(np.float32),
                scale=scale.astype(np.float32),
                quat=quat.astype(np.float32),
                n_props=len(props),
                props=props)


# ---------------------------------------------------------------------------
# Synthetic 3DGS for testing
# ---------------------------------------------------------------------------

def make_synthetic_ply(path: str, n: int = 4000, seed: int = 0) -> None:
    """Write a small 3DGS .ply of a coloured sphere shell, in the same
    format as save_gaussian_ply (17 props/vertex)."""
    rng = np.random.default_rng(seed)

    # points on a sphere shell, with two coloured caps so orientation
    # is obvious in the turntable.
    dirs = rng.normal(size=(n, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    radii = 0.30 + 0.02 * rng.standard_normal(n).astype(np.float32)
    xyz = dirs * radii[:, None]

    # colour: red top cap, green bottom cap, blue belt
    rgb = np.zeros((n, 3), dtype=np.float32)
    y = dirs[:, 1]
    top = y > 0.5
    bot = y < -0.5
    belt = ~(top | bot)
    rgb[top] = np.array([0.95, 0.20, 0.20])
    rgb[bot] = np.array([0.20, 0.95, 0.30])
    rgb[belt] = np.array([0.25, 0.40, 0.95])
    # add a subtle colour gradient along x so left/right is also distinct
    rgb[:, 0] = np.clip(rgb[:, 0] + 0.15 * dirs[:, 0], 0.0, 1.0)

    # invert the activations to recover stored values:
    SH_C0 = 0.28209479177387814
    f_dc = (rgb - 0.5) / SH_C0
    opacity_logit = np.full((n,), 2.0, dtype=np.float32)        # sigmoid(2)~0.88
    sigma = 0.012
    log_scale = np.full((n, 3), math.log(sigma), dtype=np.float32)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    normals = np.zeros_like(xyz)
    row = np.concatenate(
        [xyz, normals, f_dc, opacity_logit[:, None], log_scale, quat],
        axis=1).astype("<f4")
    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity",
             "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    assert row.shape[1] == len(props)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for p in props:
            f.write(f"property float {p}\n".encode())
        f.write(b"end_header\n")
        f.write(row.tobytes())


# ---------------------------------------------------------------------------
# Camera + math
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) (w, x, y, z) unit quats -> (N, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray
            ) -> np.ndarray:
    """Right-handed view matrix: world -> camera. Camera looks down -Z."""
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


# ---------------------------------------------------------------------------
# Splat rasterizer
# ---------------------------------------------------------------------------

def render_view(splat: dict,
                view: np.ndarray,
                width: int = 512,
                height: int = 512,
                fov_y_deg: float = 35.0,
                bg: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                ) -> np.ndarray:
    """Anisotropic Gaussian splat rasterizer.

    Returns BGR uint8 image (cv2 convention).
    """
    xyz = splat["xyz"]                   # (N, 3) world
    rgb = splat["rgb"]                   # (N, 3) [0,1]
    op = splat["opacity"]                # (N,)
    scl = splat["scale"]                 # (N, 3) sigmas
    R = quat_to_rotmat(splat["quat"])    # (N, 3, 3)

    # ---- world -> camera ----
    Rvw = view[:3, :3]
    tvw = view[:3, 3]
    cam = xyz @ Rvw.T + tvw              # (N, 3)
    z_cam = cam[:, 2]                    # camera looks down -Z, so z<0 is front
    in_front = z_cam < -1e-3
    if not np.any(in_front):
        bg_bgr = (np.array(bg)[::-1] * 255).astype(np.uint8)
        return np.broadcast_to(bg_bgr, (height, width, 3)).copy()

    # ---- perspective intrinsics ----
    fy = (height * 0.5) / math.tan(math.radians(fov_y_deg) * 0.5)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5

    # ---- project centres: u = fx*x/(-z) + cx, v = -fy*y/(-z) + cy
    #      (image y axis grows downward; world y up) -------------------------
    inv_neg_z = 1.0 / (-z_cam + 1e-8)
    u = fx * cam[:, 0] * inv_neg_z + cx
    v = -fy * cam[:, 1] * inv_neg_z + cy

    # ---- world covariance:  Σ_w = R diag(s)^2 R^T -------------------------
    s2 = scl ** 2
    sigma_w = np.einsum("nij,nj,nkj->nik", R, s2, R)            # (N, 3, 3)
    # camera covariance: Σ_c = Rvw Σ_w Rvw^T (constant Rvw -> simple)
    sigma_c = np.einsum("ij,njk,lk->nil", Rvw, sigma_w, Rvw)    # (N, 3, 3)

    # ---- 2x3 Jacobian of perspective at each centre ----------------------
    inv = inv_neg_z
    inv2 = inv * inv
    # d(u)/d(xc, yc, zc), d(v)/...
    # u = fx * xc * (-1/zc) + cx  => du/dxc = -fx/zc, du/dyc=0, du/dzc = fx*xc/zc^2
    # but we use inv = 1/(-zc), so du/dxc = fx*inv, du/dzc = fx*xc*inv2 *(sign)
    # Let's derive cleanly with z' = -zc > 0:
    # u = fx*xc/z' + cx                du/dxc = fx/z'           = fx*inv
    #                                  du/dz' = -fx*xc/z'^2     = -fx*xc*inv2
    # dz'/dzc = -1                      => du/dzc = fx*xc*inv2
    # v = -fy*yc/z' + cy                dv/dyc = -fy/z'          = -fy*inv
    #                                  dv/dz' = fy*yc/z'^2      = fy*yc*inv2
    # dv/dzc = -fy*yc*inv2
    J = np.zeros((cam.shape[0], 2, 3), dtype=np.float32)
    J[:, 0, 0] = fx * inv
    J[:, 0, 2] = fx * cam[:, 0] * inv2
    J[:, 1, 1] = -fy * inv
    J[:, 1, 2] = -fy * cam[:, 1] * inv2

    # 2x2 image covariance + small antialiasing dilation
    sigma_2d = np.einsum("nij,njk,nlk->nil", J, sigma_c, J)     # (N, 2, 2)
    sigma_2d[:, 0, 0] += 0.3
    sigma_2d[:, 1, 1] += 0.3

    det = sigma_2d[:, 0, 0] * sigma_2d[:, 1, 1] - sigma_2d[:, 0, 1] ** 2
    valid = in_front & (det > 1e-6) & np.isfinite(u) & np.isfinite(v)

    # 3-sigma bbox half-extent in each axis from eigenvalues
    a = sigma_2d[:, 0, 0]
    c = sigma_2d[:, 1, 1]
    b = sigma_2d[:, 0, 1]
    tr = a + c
    disc = np.sqrt(np.maximum(0.25 * (a - c) ** 2 + b ** 2, 0.0))
    lam1 = 0.5 * tr + disc
    radius = 3.0 * np.sqrt(np.maximum(lam1, 1e-8))

    # cull splats whose 3-sigma circle is entirely off-screen / too tiny
    valid &= (radius > 0.5) & (radius < max(width, height) * 0.6)
    valid &= (u + radius > 0) & (u - radius < width)
    valid &= (v + radius > 0) & (v - radius < height)

    idx = np.where(valid)[0]
    if idx.size == 0:
        bg_bgr = (np.array(bg)[::-1] * 255).astype(np.uint8)
        return np.broadcast_to(bg_bgr, (height, width, 3)).copy()

    # Front-to-back compositing -> render order = farthest first.
    # Camera looks down -Z so larger z_cam (=closer to 0) means closer.
    order = idx[np.argsort(z_cam[idx])]      # most negative z first = farthest

    # Inverse 2D covariance for the gaussian eval.
    inv_det = 1.0 / det
    inv_a = c * inv_det
    inv_c = a * inv_det
    inv_b = -b * inv_det

    # Output buffers
    img = np.zeros((height, width, 3), dtype=np.float32)
    alpha_acc = np.zeros((height, width), dtype=np.float32)

    # For each splat in back-to-front order, blend into the canvas.
    # OVER operator with an "accumulated foreground alpha" buffer, then
    # composite over background at the end (front-to-back would let us
    # early-out, but back-to-front is simpler & we're CPU-bound on
    # per-splat python overhead either way).
    rgb_u = rgb[order]
    op_u = op[order]
    u_u = u[order]
    v_u = v[order]
    r_u = radius[order]
    ia = inv_a[order]
    ic = inv_c[order]
    ib = inv_b[order]

    for k in range(order.size):
        cu, cv = u_u[k], v_u[k]
        rad = r_u[k]
        x0 = max(int(math.floor(cu - rad)), 0)
        x1 = min(int(math.ceil(cu + rad)) + 1, width)
        y0 = max(int(math.floor(cv - rad)), 0)
        y1 = min(int(math.ceil(cv + rad)) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1, dtype=np.float32) - cu
        ys = np.arange(y0, y1, dtype=np.float32) - cv
        XX, YY = np.meshgrid(xs, ys)
        # quadratic form: 0.5 * [x y] Σ⁻¹ [x y]^T
        q = 0.5 * (ia[k] * XX * XX
                   + 2 * ib[k] * XX * YY
                   + ic[k] * YY * YY)
        np.maximum(q, 0.0, out=q)
        # cap to keep exp tractable; 12 ~ ~1e-5 weight
        np.minimum(q, 12.0, out=q)
        w = np.exp(-q) * op_u[k]
        # OVER:  c_out = c_in*(1-a_dst) + c_src*a_src*?  We accumulate
        # premultiplied colour and alpha, so:
        # new_alpha = a_dst + (1-a_dst)*w
        # new_color = c_dst + (1-a_dst)*w*rgb_src
        a_dst = alpha_acc[y0:y1, x0:x1]
        one_minus = 1.0 - a_dst
        contrib = one_minus * w
        alpha_acc[y0:y1, x0:x1] = a_dst + contrib
        img[y0:y1, x0:x1, 0] += contrib * rgb_u[k, 0]
        img[y0:y1, x0:x1, 1] += contrib * rgb_u[k, 1]
        img[y0:y1, x0:x1, 2] += contrib * rgb_u[k, 2]

    # Composite over background.
    bg_arr = np.array(bg, dtype=np.float32)
    img += (1.0 - alpha_acc)[:, :, None] * bg_arr[None, None, :]
    img = np.clip(img, 0.0, 1.0)

    # RGB -> BGR for cv2
    bgr = (img[:, :, ::-1] * 255.0).astype(np.uint8)
    return bgr


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _camera_for_angle(scene_center: np.ndarray, scene_radius: float,
                      angle_rad: float, elevation_deg: float = 15.0,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Build a view matrix orbiting around scene_center at radius ~3*scene_radius."""
    cam_radius = max(scene_radius, 1e-3) * 3.0
    el = math.radians(elevation_deg)
    eye = scene_center + np.array([
        cam_radius * math.cos(el) * math.sin(angle_rad),
        cam_radius * math.sin(el),
        cam_radius * math.cos(el) * math.cos(angle_rad),
    ], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return look_at(eye, scene_center, up), eye


def render_turntable(splat: dict,
                     out_mp4: str,
                     width: int = 512,
                     height: int = 512,
                     n_frames: int = 60,
                     fps: int = 30,
                     fov_y_deg: float = 35.0,
                     ) -> None:
    # Scene bounds — centre on weighted opacity to avoid stragglers.
    op = splat["opacity"]
    w = op / (op.sum() + 1e-8)
    centre = (splat["xyz"] * w[:, None]).sum(axis=0)
    radius = float(np.linalg.norm(splat["xyz"] - centre, axis=1).max())

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed for {out_mp4}")
    try:
        for i in range(n_frames):
            ang = 2.0 * math.pi * i / n_frames
            view, _ = _camera_for_angle(centre, radius, ang)
            frame = render_view(splat, view,
                                width=width, height=height,
                                fov_y_deg=fov_y_deg)
            vw.write(frame)
    finally:
        vw.release()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Quick CPU preview for a 3DGS .ply (Option B).")
    p.add_argument("ply", nargs="?", default=None,
                   help="path to splat.ply (omit with --synthetic to "
                        "auto-generate one in /tmp).")
    p.add_argument("--synthetic", action="store_true",
                   help="generate a synthetic 3DGS .ply for testing.")
    p.add_argument("--out", default=None,
                   help="output dir (default: alongside .ply).")
    p.add_argument("--size", type=int, default=512,
                   help="render WxH (default 512).")
    p.add_argument("--frames", type=int, default=60,
                   help="turntable frame count (default 60).")
    p.add_argument("--fps", type=int, default=30, help="mp4 fps (default 30).")
    p.add_argument("--fov", type=float, default=35.0,
                   help="vertical FOV deg (default 35).")
    p.add_argument("--no-mp4", action="store_true",
                   help="only write the front-view PNG.")
    args = p.parse_args(argv)

    if args.synthetic and args.ply is None:
        args.ply = "/tmp/synthetic_splat.ply"
        print(f"[synthetic] writing random splat to {args.ply}")
        make_synthetic_ply(args.ply, n=4000)
    if args.ply is None:
        p.error("supply a .ply path or use --synthetic")

    ply_path = Path(args.ply)
    if not ply_path.is_file():
        print(f"[error] {ply_path} not found", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else ply_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ply_path.stem

    t0 = time.time()
    splat = load_3dgs_ply(str(ply_path))
    n = splat["xyz"].shape[0]
    print(f"[load] {n} gaussians, {splat['n_props']} props/vertex "
          f"({time.time() - t0:.2f}s)")

    # Scene bounds
    op = splat["opacity"]
    w = op / (op.sum() + 1e-8)
    centre = (splat["xyz"] * w[:, None]).sum(axis=0)
    radius = float(np.linalg.norm(splat["xyz"] - centre, axis=1).max())
    print(f"[scene] centre={centre}, radius={radius:.4f}")

    # Front view
    t0 = time.time()
    view, _ = _camera_for_angle(centre, radius, angle_rad=0.0,
                                elevation_deg=15.0)
    img = render_view(splat, view,
                      width=args.size, height=args.size, fov_y_deg=args.fov)
    png_path = out_dir / f"{stem}_preview.png"
    cv2.imwrite(str(png_path), img)
    print(f"[png ] {png_path} ({time.time() - t0:.2f}s)")

    if not args.no_mp4:
        t0 = time.time()
        mp4_path = out_dir / f"{stem}_turntable.mp4"
        render_turntable(splat, str(mp4_path),
                         width=args.size, height=args.size,
                         n_frames=args.frames, fps=args.fps,
                         fov_y_deg=args.fov)
        print(f"[mp4 ] {mp4_path}  ({args.frames} frames, "
              f"{time.time() - t0:.2f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
