"""Apply an iridescent / thin-film / holographic-foil look to a 3DGS .ply.

Single-pass post-processing — reads an input `.ply`, computes a synthetic
view-dependent rainbow colour per Gaussian, writes the result. Works
because Gaussians have an implicit local frame (rotation quaternion +
3-axis scale) from which we can recover a *normal*: the principal axis
along the smallest scale direction.

The colour model is a mash-up of:
  1. **Thin-film hue band** — `hue = (|n·v| · freq + phase) mod 1`
  2. **Fresnel-style metallic base** — high at grazing angles
  3. **White highlight** — high near n·v = 1

Output `.ply` is bit-identical to the input except for the three
`f_dc_*` SH-DC fields. Opacities, scales, positions, rotations are all
preserved → existing viewers (SuperSplat / WebGL splat viewer) render
the recoloured Gaussians correctly.

For animation: sweep `--phase` from 0 to 1 across N output PLYs, render
each as a thumbnail or 360° GIF, stitch frames.

Usage:
    python meadow_wb/scripts/apply_iridescent.py input.ply iridescent.ply \\
        --view 0 0 1 --freq 4.0 --phase 0.0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814  # Y_0^0; rgb = clip(0.5 + SH_C0 * f_dc, 0, 1)


def _read_ply(path: Path):
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


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from (N, 4) quaternions stored as (w, x, y, z)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _hue_to_rgb(hue: np.ndarray) -> np.ndarray:
    """Smooth rainbow ramp h in [0, 1] → RGB in [0, 1] (sin-based, not HSV)."""
    h = hue * 2.0 * math.pi
    r = 0.5 + 0.5 * np.sin(h)
    g = 0.5 + 0.5 * np.sin(h + 2.0 * math.pi / 3.0)
    b = 0.5 + 0.5 * np.sin(h + 4.0 * math.pi / 3.0)
    return np.stack([r, g, b], axis=-1)


def apply_iridescent(
    in_path: str,
    out_path: str,
    view: np.ndarray,
    freq: float,
    phase: float,
    metallic_mix: float,
    normal_source: str,
):
    header, body = _read_ply(Path(in_path))
    n = len(body)

    if normal_source == "quat":
        # Build per-Gaussian world-space normal from quaternion + smallest scale.
        # Works only if the Gaussians are visibly anisotropic; with the
        # SLAT-POLISH 0.010 scale clamp our outputs are near-isotropic and
        # this path becomes noisy.
        scales = np.stack([body["scale_0"], body["scale_1"], body["scale_2"]], axis=-1)
        sigma = np.exp(scales)
        quats = np.stack([body["rot_0"], body["rot_1"], body["rot_2"], body["rot_3"]], axis=-1)
        R = _quat_to_matrix(quats.astype(np.float32))
        smallest_axis = np.argmin(sigma, axis=-1)
        local_normal = np.zeros((n, 3), dtype=np.float32)
        local_normal[np.arange(n), smallest_axis] = 1.0
        world_normal = np.einsum("nij,nj->ni", R, local_normal)
    else:  # "radial" — smooth pseudo-normal from object centre outward
        xyz = np.stack([body["x"], body["y"], body["z"]], axis=-1).astype(np.float32)
        centre = (xyz.max(0) + xyz.min(0)) * 0.5
        world_normal = xyz - centre
    world_normal /= np.linalg.norm(world_normal, axis=-1, keepdims=True).clip(min=1e-6)

    v = view / np.linalg.norm(view)
    # |n·v| in [0, 1].
    cos_theta = np.abs(world_normal @ v)

    # Iridescent hue band.
    hue = (cos_theta * freq + phase) % 1.0
    rainbow = _hue_to_rgb(hue)

    # Two presets controlled by `metallic_mix` (0 = candy, 1 = chrome):
    # candy   = full rainbow + soft silver mix
    # chrome  = dark chrome base, rainbow appears only at grazing band
    metallic_base = np.array([0.78, 0.80, 0.86])  # cooler darker silver
    # Edge-band: pop rainbow only where n·v ≈ a few critical angles.
    edge_band = np.exp(-((cos_theta - 0.55) ** 2) / 0.025)  # ring at mid-angle
    rim_band  = np.exp(-((cos_theta - 0.05) ** 2) / 0.01)   # narrow grazing band
    grazing_w = np.clip(edge_band + rim_band, 0.0, 1.0)[:, None]
    chrome = (1.0 - grazing_w) * metallic_base + grazing_w * rainbow

    # Specular highlight near facing.
    spec = cos_theta[:, None] ** 16
    chrome = np.clip(chrome + 0.5 * spec, 0.0, 1.0)

    # Blend: 0 = pure rainbow (candy), 1 = chrome look
    candy = 0.55 * rainbow + 0.45 * (rainbow + 0.1)
    rgb = (1.0 - metallic_mix) * candy + metallic_mix * chrome

    # Convert rgb -> SH DC.
    f_dc = (rgb - 0.5) / SH_C0

    out = body.copy()
    out["f_dc_0"] = f_dc[:, 0].astype(np.float32)
    out["f_dc_1"] = f_dc[:, 1].astype(np.float32)
    out["f_dc_2"] = f_dc[:, 2].astype(np.float32)

    with open(out_path, "wb") as f:
        f.write(header)
        out.tofile(f)
    print(f"[iri] {out_path}  N={n}  freq={freq}  phase={phase:.3f}  view={view.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--view", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                    help="Fixed eye direction in world space (will be normalised)")
    ap.add_argument("--freq", type=float, default=4.0,
                    help="Hue cycles across n·v ∈ [0,1]")
    ap.add_argument("--phase", type=float, default=0.0,
                    help="Hue phase offset in [0,1]; sweep this for animation")
    ap.add_argument("--metallic-mix", type=float, default=0.55,
                    help="0 = pure rainbow, 1 = full Fresnel-metallic mix")
    ap.add_argument("--normal", choices=["radial", "quat"], default="radial",
                    help="'radial' (smooth, default): pseudo-normal from object centre. "
                         "'quat': from Gaussian rotation + scale (noisy if scales near-isotropic).")
    args = ap.parse_args()
    apply_iridescent(
        args.inp, args.out,
        view=np.array(args.view, dtype=np.float32),
        freq=args.freq,
        phase=args.phase,
        metallic_mix=args.metallic_mix,
        normal_source=args.normal,
    )
