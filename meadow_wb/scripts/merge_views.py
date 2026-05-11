"""Multi-view 3D Gaussian Splat merger.

Takes N `.ply` files (each a single-view Meadow World Builder output),
estimates pairwise relative poses via Open3D's tensor-API ICP with a
small grid of rotation hypotheses (poor-man's global init — Open3D's
legacy RANSAC API segfaults on Apple-Silicon arm64 0.18), transforms
every view into the first view's coordinate frame, and concatenates the
Gaussians into one output `.ply`.

Honest limitations — single-image 3DGS outputs are in *object-centric*
frames whose orientation depends on the input camera angle. Two views
of the same object captured from very different angles are unlikely to
register cleanly via point-only ICP; expect this prototype to work best
when:

* the two views differ by a roughly known coarse rotation (~< 90°), or
* you supply a `--rotation-hypotheses` grid that covers the expected
  azimuth range.

The proper long-term fix is image-side pose estimation
(DUSt3R / MASt3R) feeding poses straight into the merger. See the
report at `docs/REPORT_MULTI_VIEW_MERGE.md` for ablations and the
real-input failure mode.

Usage:
    python meadow_wb/scripts/merge_views.py \
        view_a.ply view_b.ply view_c.ply \
        --out merged.ply \
        --voxel-size 0.02 \
        --rotation-hypotheses 8     # try 0°, 45°, 90°, ... around Y as init
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(
        "open3d is not installed. Add it to your venv with `pip install open3d` "
        "before running this script (not pinned in requirements.txt because the "
        "merge workflow is opt-in and the wheel is ~300 MB)."
    ) from exc


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def _read_ply(path: Path):
    """Return (header_bytes, structured_array)."""
    with open(path, "rb") as f:
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
        dtype = np.dtype([(name, dm[t]) for t, name in props])
        body = np.fromfile(f, dtype=dtype, count=n)
    return header, body


def _to_tensor_pcd(arr, voxel_size: float):
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(xyz)
    pcd_ds = pcd.voxel_down_sample(voxel_size)
    pcd_ds.estimate_normals(radius=voxel_size * 2, max_nn=30)
    return pcd_ds


# ---------------------------------------------------------------------------
# Multi-start ICP (replaces RANSAC for ARM-safe global init)
# ---------------------------------------------------------------------------

def _yaw_T(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return T


def _multi_start_icp(
    src_pcd,
    dst_pcd,
    voxel_size: float,
    n_hypotheses: int,
):
    """Try `n_hypotheses` Y-axis rotations as init, return the best ICP fit.

    A poor-man's global registration that's stable on Apple-Silicon Open3D.
    """
    best = None
    distance_threshold_coarse = voxel_size * 4.0
    distance_threshold_fine = voxel_size * 0.4
    for i in range(n_hypotheses):
        ang = 2.0 * math.pi * i / n_hypotheses
        init = o3d.core.Tensor(_yaw_T(ang), dtype=o3d.core.float64)
        # Coarse ICP (large correspondence distance) to escape bad init.
        coarse = o3d.t.pipelines.registration.icp(
            src_pcd,
            dst_pcd,
            max_correspondence_distance=distance_threshold_coarse,
            init_source_to_target=init,
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        # Fine ICP from the coarse result.
        fine = o3d.t.pipelines.registration.icp(
            src_pcd,
            dst_pcd,
            max_correspondence_distance=distance_threshold_fine,
            init_source_to_target=coarse.transformation,
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        if best is None or fine.fitness > best.fitness:
            best = fine
    return best


# ---------------------------------------------------------------------------
# Apply 4x4 transform to a structured PLY array (xyz + Gaussian quaternion)
# ---------------------------------------------------------------------------

def _apply_T(arr, T):
    out = arr.copy()
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    new_xyz = xyz @ R.T + t
    out["x"] = new_xyz[:, 0]
    out["y"] = new_xyz[:, 1]
    out["z"] = new_xyz[:, 2]

    quat_fields = [f for f in ("rot_0", "rot_1", "rot_2", "rot_3") if f in arr.dtype.names]
    if len(quat_fields) == 4:
        Rq = _R_to_quat(R)
        q_old = np.stack([arr[f] for f in quat_fields], axis=1).astype(np.float32)
        q_new = _quat_mul(Rq[None, :], q_old)
        for i, f in enumerate(quat_fields):
            out[f] = q_new[:, i]
    return out


def _R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s], dtype=np.float32)
    diag = (R[0, 0], R[1, 1], R[2, 2])
    i = int(np.argmax(diag))
    if i == 0:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s], dtype=np.float32)
    if i == 1:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s], dtype=np.float32)
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s], dtype=np.float32)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge(paths, out_path, voxel_size, n_hypotheses):
    print(f"[merge] {len(paths)} views, voxel={voxel_size}, hypotheses={n_hypotheses}")
    headers, bodies, pcds = [], [], []
    for p in paths:
        h, body = _read_ply(Path(p))
        headers.append(h)
        bodies.append(body)
        pcd = _to_tensor_pcd(body, voxel_size)
        pcds.append(pcd)
        print(f"  {p}: {len(body)} Gaussians → {len(pcd.point.positions)} after downsample")

    transforms = [np.eye(4)]
    for i in range(1, len(paths)):
        t0 = time.time()
        result = _multi_start_icp(pcds[i], pcds[0], voxel_size, n_hypotheses)
        T = result.transformation.numpy()
        transforms.append(T)
        print(
            f"  view {i} → view 0:  fitness={result.fitness:.3f}  "
            f"inlier_rmse={result.inlier_rmse:.4f}  "
            f"({time.time()-t0:.1f}s, {n_hypotheses} starts)"
        )

    transformed = [_apply_T(bodies[i], transforms[i]) for i in range(len(paths))]
    merged = np.concatenate(transformed, axis=0)

    out_header = b""
    for line in headers[0].split(b"\n"):
        if line.startswith(b"element vertex"):
            out_header += f"element vertex {len(merged)}".encode() + b"\n"
        else:
            out_header += line + b"\n"
    out_header = out_header.rstrip(b"\n") + b"\n"

    with open(out_path, "wb") as f:
        f.write(out_header)
        merged.tofile(f)
    print(f"[merge] {out_path}: {len(merged)} Gaussians ({Path(out_path).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plys", nargs="+", help="Input .ply files (first one is the reference frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel-size", type=float, default=0.02)
    ap.add_argument("--rotation-hypotheses", type=int, default=8,
                    help="N yaw-rotation initial guesses to try (cheap RANSAC replacement)")
    args = ap.parse_args()
    if len(args.plys) < 2:
        raise SystemExit("Need at least 2 input PLYs to merge.")
    merge(args.plys, args.out, args.voxel_size, args.rotation_hypotheses)
