"""ply_to_spz.py - Convert a 3DGS .ply (SAM 3D Objects output) to Niantic .spz.

Usage:
    python meadow3d/scripts/ply_to_spz.py splat.ply [--out splat.spz] [--sh-degree 0]
    python meadow3d/scripts/ply_to_spz.py --check splat.spz   # roundtrip stats

Approach (Option C in SPEC_SPZ_EXPORT.md):
    Use the `spz` PyPI package (Rust/PyO3 bindings around Niantic's reference
    crate). We parse the PLY ourselves (no plyfile dep), pull the canonical
    3DGS field set, and hand it straight to spz.GaussianSplat.save() with
    the source coordinate system set to RDF (the convention Niantic uses
    for raw PLY files).

Field mapping (matches Niantic's loadSplatFromPly):
    PLY  x,y,z          -> positions
    PLY  scale_0..2     -> scales        (already log-scale; no transform)
    PLY  rot_0..3       -> rotations     ((w,x,y,z); spz normalizes internally)
    PLY  opacity        -> alphas        (already inverse-sigmoid)
    PLY  f_dc_0..2      -> colors        (raw SH degree-0 coefficients)
    PLY  f_rest_*       -> spherical_harmonics (only if --sh-degree > 0)

Notes:
    * We do NOT modify sam3d_objects/ - this is a utility tool.
    * The spz PyPI package (v0.0.1) ships a packaging bug where
      spz/__init__.py imports `from spz import ...` instead of
      `from spz.spz import ...`. The fix is one line; if the import below
      fails on a clean install, patch that file.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import spz  # type: ignore
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import `spz` (PyPI package).\n"
        "  pip install spz\n"
        "If that succeeds but `import spz` still fails with a circular import,\n"
        "the wheel has a known packaging bug: edit\n"
        "  <site-packages>/spz/__init__.py\n"
        "and change `from spz import (...)` to `from spz.spz import (...)`.\n"
        f"Original error: {e}"
    )


# ---------------------------------------------------------------------------
# Minimal binary-little-endian PLY parser for 3DGS files
# ---------------------------------------------------------------------------

def _parse_ply_header(fh) -> Tuple[int, List[str]]:
    """Return (num_vertices, property_names).

    Only supports binary_little_endian float32 vertex elements - which is
    exactly what save_gaussian_ply emits. We bail otherwise so the user
    knows to convert their PLY first.
    """
    line = fh.readline().rstrip(b"\n").rstrip(b"\r")
    if line != b"ply":
        raise ValueError(f"not a PLY file (magic = {line!r})")
    fmt = fh.readline().rstrip(b"\n").rstrip(b"\r")
    if fmt != b"format binary_little_endian 1.0":
        raise ValueError(
            f"only binary_little_endian PLY is supported, got: {fmt!r}"
        )

    num_vertices = -1
    props: List[str] = []
    in_vertex = False
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("unexpected EOF in PLY header")
        line = line.rstrip(b"\n").rstrip(b"\r")
        if line == b"end_header":
            break
        if line.startswith(b"comment"):
            continue
        if line.startswith(b"element"):
            parts = line.split()
            if parts[1] == b"vertex":
                num_vertices = int(parts[2])
                in_vertex = True
            else:
                in_vertex = False
            continue
        if line.startswith(b"property") and in_vertex:
            parts = line.split()
            # property <type> <name>
            if parts[1] != b"float" and parts[1] != b"float32":
                raise ValueError(
                    f"only float32 vertex properties supported, got {parts!r}"
                )
            props.append(parts[2].decode("ascii"))

    if num_vertices < 0 or not props:
        raise ValueError("PLY header missing vertex element or properties")
    return num_vertices, props


def load_3dgs_ply(path: str) -> Tuple[Dict[str, np.ndarray], int]:
    """Load a binary-little-endian 3DGS .ply.

    Returns:
        fields: dict mapping each property name -> (N,) float32 ndarray
        sh_degree: 0/1/2/3 inferred from f_rest_* count
                   (0 => 0, 9 => 1, 24 => 2, 45 => 3)
    """
    with open(path, "rb") as fh:
        n, props = _parse_ply_header(fh)
        raw = np.frombuffer(fh.read(n * len(props) * 4), dtype="<f4")
    if raw.size != n * len(props):
        raise ValueError(
            f"PLY size mismatch: expected {n * len(props)} floats, got {raw.size}"
        )
    table = raw.reshape(n, len(props))
    fields = {name: table[:, i].copy() for i, name in enumerate(props)}

    n_rest = sum(1 for k in props if k.startswith("f_rest_"))
    if n_rest == 0:
        sh_degree = 0
    elif n_rest == 9:
        sh_degree = 1
    elif n_rest == 24:
        sh_degree = 2
    elif n_rest == 45:
        sh_degree = 3
    else:
        raise ValueError(
            f"f_rest_* count = {n_rest} doesn't match SH degree 0/1/2/3"
        )
    return fields, sh_degree


# ---------------------------------------------------------------------------
# PLY -> SPZ
# ---------------------------------------------------------------------------

def _stack(fields: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([fields[n] for n in names], axis=1).astype(np.float32)
    )


def ply_to_spz(
    ply_path: str,
    spz_path: str,
    sh_degree_override: int | None = None,
    coord_system: str = "RDF",
) -> dict:
    """Convert .ply -> .spz, return a stats dict."""
    t0 = time.time()
    fields, sh_degree = load_3dgs_ply(ply_path)
    if sh_degree_override is not None:
        if sh_degree_override > sh_degree:
            raise ValueError(
                f"--sh-degree {sh_degree_override} > inferred {sh_degree} "
                f"from f_rest count"
            )
        sh_degree = sh_degree_override
    n = next(iter(fields.values())).shape[0]
    t_load = time.time() - t0

    positions = _stack(fields, ["x", "y", "z"])
    scales = _stack(fields, ["scale_0", "scale_1", "scale_2"])
    rotations = _stack(fields, ["rot_0", "rot_1", "rot_2", "rot_3"])  # w,x,y,z
    alphas = fields["opacity"].astype(np.float32, copy=True)
    colors = _stack(fields, ["f_dc_0", "f_dc_1", "f_dc_2"])

    sh: np.ndarray | None = None
    if sh_degree > 0:
        sh_dim = {1: 3, 2: 8, 3: 15}[sh_degree]
        # PLY stores f_rest in [coeff, channel] order: f_rest_0..f_rest_{3*sh_dim-1}
        # where the first sh_dim values are R coeffs, then G, then B.
        # spz expects (N, sh_dim, 3) flattened to (N, sh_dim*3) with the inner
        # dim = (R,G,B) per coefficient. Re-interleave.
        rest_flat = _stack(fields, [f"f_rest_{i}" for i in range(3 * sh_dim)])
        # rest_flat is (N, 3*sh_dim) with layout [R0..R{sh_dim-1}, G0..., B0...]
        # Reshape to (N, 3, sh_dim) then transpose to (N, sh_dim, 3).
        sh = (
            rest_flat.reshape(n, 3, sh_dim)
            .transpose(0, 2, 1)
            .reshape(n, sh_dim * 3)
            .astype(np.float32, copy=False)
        )
        sh = np.ascontiguousarray(sh)

    splat = spz.GaussianSplat(
        positions=positions,
        scales=scales,
        rotations=rotations,
        alphas=alphas,
        colors=colors,
        sh_degree=sh_degree,
        spherical_harmonics=sh,
        antialiased=False,
    )

    cs = getattr(spz.CoordinateSystem, coord_system)
    Path(spz_path).parent.mkdir(parents=True, exist_ok=True)
    splat.save(spz_path, from_coordinate_system=cs)
    t_total = time.time() - t0

    ply_size = os.path.getsize(ply_path)
    spz_size = os.path.getsize(spz_path)
    return {
        "n_gaussians": n,
        "sh_degree": sh_degree,
        "ply_bytes": ply_size,
        "spz_bytes": spz_size,
        "ratio": ply_size / max(spz_size, 1),
        "load_s": t_load,
        "total_s": t_total,
    }


# ---------------------------------------------------------------------------
# Sanity check (decode roundtrip)
# ---------------------------------------------------------------------------

def check_spz(spz_path: str, ply_path: str | None = None) -> None:
    """Decode the SPZ and print summary stats. If a .ply is given, compare."""
    splat = spz.load(spz_path)
    print(f"[check] {spz_path}")
    print(f"        num_points = {splat.num_points}")
    print(f"        sh_degree  = {splat.sh_degree}")
    print(f"        bbox       = {splat.bbox}")
    pos = splat.positions
    print(
        f"        pos range  x[{pos[:,0].min():+.3f}, {pos[:,0].max():+.3f}] "
        f"y[{pos[:,1].min():+.3f}, {pos[:,1].max():+.3f}] "
        f"z[{pos[:,2].min():+.3f}, {pos[:,2].max():+.3f}]"
    )

    if ply_path is None:
        return
    fields, _ = load_3dgs_ply(ply_path)
    src = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)
    # spz default coord system is UNSPECIFIED (no flip), but the encoded file
    # was written from RDF, so loading it back with UNSPECIFIED returns RUB
    # internal storage. To compare apples to apples, reload with from->RDF.
    splat_rdf = spz.load(spz_path, coordinate_system=spz.CoordinateSystem.RDF)
    pos_rdf = splat_rdf.positions
    n = min(src.shape[0], pos_rdf.shape[0])
    diff = pos_rdf[:n] - src[:n]
    rms = float(np.sqrt((diff ** 2).mean()))
    maxabs = float(np.abs(diff).max())
    print(
        f"[check] roundtrip vs PLY positions: "
        f"rms={rms:.5g}, max|err|={maxabs:.5g}  "
        f"(quantized to fixed-point inside SPZ - small diffs expected)"
    )

    # Color & alpha roundtrip (should be lossy due to u8 quantization).
    src_color = np.stack(
        [fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]], axis=1
    )
    col = splat_rdf.colors
    diff_c = np.abs(col[:n] - src_color[:n]).mean()
    diff_a = np.abs(splat_rdf.alphas[:n] - fields["opacity"][:n]).mean()
    print(f"[check] color  mean|err| = {diff_c:.5g}")
    print(f"[check] alpha  mean|err| = {diff_a:.5g}")


# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a 3DGS .ply to Niantic .spz."
    )
    ap.add_argument("ply", help="input .ply path (binary little-endian 3DGS)")
    ap.add_argument(
        "--out", default=None,
        help="output .spz path (default: same stem as input)",
    )
    ap.add_argument(
        "--sh-degree", type=int, default=None,
        help="override SH degree (default: auto from f_rest count)",
    )
    ap.add_argument(
        "--coord", default="RDF",
        choices=["RDF", "RUB", "RUF", "RDB", "LDB", "LDF", "LUB", "LUF",
                 "UNSPECIFIED"],
        help="source coordinate system of the PLY (default: RDF, "
             "which is the niantic reference convention)",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="after writing, decode .spz and print roundtrip stats",
    )
    args = ap.parse_args(argv)

    out = args.out or str(Path(args.ply).with_suffix(".spz"))
    stats = ply_to_spz(
        args.ply, out,
        sh_degree_override=args.sh_degree,
        coord_system=args.coord,
    )
    print(
        f"[ply_to_spz] {args.ply}  ->  {out}\n"
        f"             {stats['n_gaussians']:>8d} Gaussians, "
        f"sh_degree={stats['sh_degree']}\n"
        f"             {stats['ply_bytes']/1e6:8.2f} MB  ->  "
        f"{stats['spz_bytes']/1e6:6.2f} MB   "
        f"({stats['ratio']:.2f}x compression)\n"
        f"             elapsed {stats['total_s']:.2f}s "
        f"(parse {stats['load_s']:.2f}s)"
    )

    if args.check:
        check_spz(out, args.ply)

    return 0


if __name__ == "__main__":
    sys.exit(main())
