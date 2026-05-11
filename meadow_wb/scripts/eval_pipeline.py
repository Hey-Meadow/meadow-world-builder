"""Evaluation harness for the SAM-3D Objects MLX pipeline.

This is a USER tool that drives ``SAM3DObjectsPipeline`` over a list of
``(image, mask)`` pairs and produces aggregate timing / size / status stats.
Works today against a stub pipeline so it can be tested before
``meadow_wb/models/pipeline_mlx.py`` (Agent OBJ-INTEG) is ready. When the
real pipeline lands, swap the stub with the import and rerun.

Usage
-----
    python meadow_wb/scripts/eval_pipeline.py \\
        --image-dir notebook/images \\
        --out-dir   meadow_wb/_eval_out \\
        --seeds     42,43 \\
        --max-cases 6 \\
        --pipeline  stub      # or 'real' when OBJ-INTEG is wired

Outputs
-------
    OUT/case_{name}_seed_{s}/splat.ply         (one .ply per case)
    OUT/eval_summary.csv                        (flat per-row summary)
    OUT/eval_summary.json                       (structured + aggregates)

Columns of eval_summary.csv
---------------------------
    case, image_path, mask_path, seed, status, error,
    total_sec, ss_sec, slat_sec, decode_sec,
    n_gaussians, n_voxels, ply_bytes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Repo root on path so ``meadow_wb.*`` imports resolve.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Stub pipeline (interface mirrors meadow_wb/docs/SPEC_INTEG.md).
# ---------------------------------------------------------------------------


@dataclass
class StubConfig:
    """Tunable knobs for the mock pipeline (timing, sizes, failure injection)."""

    ss_sec: float = 0.05
    slat_sec: float = 0.10
    decode_sec: float = 0.02
    n_voxels: int = 1024
    gaussians_per_voxel: int = 32
    fail_on_cases: Tuple[str, ...] = ()  # case names that should raise


class StubPipeline:
    """Mock implementation of the future ``SAM3DObjectsPipeline``.

    Returns a dict shaped like the spec: ``voxels``, ``gs_params`` (numpy here,
    real pipeline returns ``mx.array``), ``ply_bytes`` (raw bytes), and a
    ``timings`` sub-dict so the harness can record per-stage cost without
    needing a profiler.
    """

    def __init__(self, cfg: Optional[StubConfig] = None) -> None:
        self.cfg = cfg or StubConfig()

    @classmethod
    def from_npz_dir(cls, npz_dir: str) -> "StubPipeline":  # noqa: ARG003
        return cls()

    def __call__(
        self,
        image: np.ndarray,
        seed: int = 42,
        case_name: str = "",
        **_: Any,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        if case_name in cfg.fail_on_cases:
            raise RuntimeError(f"stub forced failure for case {case_name!r}")

        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        time.sleep(cfg.ss_sec)
        # seed-dependent voxel count: deterministic per seed but varies a bit
        n_vox = int(cfg.n_voxels + (rng.integers(-128, 128)))
        n_vox = max(1, n_vox)
        voxels = rng.integers(0, 64, size=(n_vox, 3)).astype(np.int32)
        t_ss = time.perf_counter() - t0

        t0 = time.perf_counter()
        time.sleep(cfg.slat_sec)
        n_gauss = n_vox * cfg.gaussians_per_voxel
        gs_params = {
            "_xyz": rng.random((n_gauss, 3), dtype=np.float32),
            "_features_dc": rng.random((n_gauss, 1, 3), dtype=np.float32),
            "_scaling": rng.standard_normal((n_gauss, 3)).astype(np.float32),
            "_rotation": rng.standard_normal((n_gauss, 4)).astype(np.float32),
            "_opacity": rng.standard_normal((n_gauss, 1)).astype(np.float32),
        }
        t_slat = time.perf_counter() - t0

        t0 = time.perf_counter()
        time.sleep(cfg.decode_sec)
        ply_bytes = _fake_ply_bytes(n_gauss, seed)
        t_dec = time.perf_counter() - t0

        return {
            "voxels": voxels,
            "gs_params": gs_params,
            "ply_bytes": ply_bytes,
            "timings": {"ss_sec": t_ss, "slat_sec": t_slat, "decode_sec": t_dec},
        }


def _fake_ply_bytes(n_gauss: int, seed: int) -> bytes:
    """Tiny binary blob that varies with seed + count, useful for size stats."""
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_gauss}\nproperty float x\nproperty float y\n"
        f"property float z\nend_header\n"
    ).encode()
    rng = np.random.default_rng(seed)
    body = rng.standard_normal((n_gauss, 3)).astype(np.float32).tobytes()
    return header + body


# ---------------------------------------------------------------------------
# Pipeline factory: stub today, real pipeline once OBJ-INTEG lands.
# ---------------------------------------------------------------------------


def load_pipeline(kind: str, npz_dir: Optional[str]):
    """Return a pipeline instance with a uniform __call__(image, seed=...) API.

    ``kind`` in {"stub", "real"}. The "real" branch is a single-line swap
    once ``meadow_wb/models/pipeline_mlx.py`` is in place.
    """
    if kind == "stub":
        return StubPipeline()
    if kind == "real":
        # Lazy import so missing deps don't break stub runs.
        from meadow_wb.models.pipeline_mlx import SAM3DObjectsPipeline  # type: ignore

        return SAM3DObjectsPipeline.from_npz_dir(
            npz_dir or "meadow_wb/weights/sam3d_objects/"
        )
    raise ValueError(f"unknown pipeline kind: {kind!r}")


# ---------------------------------------------------------------------------
# Discovery: find image.png + numbered mask siblings.
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    image_path: Path
    mask_path: Path
    mask_index: int


def discover_cases(image_dir: Path, max_cases: Optional[int] = None) -> List[Case]:
    """Walk ``image_dir`` for ``image.png`` files and pair each with sibling
    numbered ``*.png`` masks. Sorted by (scene_dir, mask_index).
    """
    cases: List[Case] = []
    for image_png in sorted(image_dir.rglob("image.png")):
        scene = image_png.parent
        scene_name = scene.name
        for mask_png in sorted(scene.glob("*.png")):
            if mask_png.name == "image.png":
                continue
            stem = mask_png.stem
            if not stem.isdigit():
                continue
            idx = int(stem)
            cases.append(
                Case(
                    name=f"{scene_name}__mask{idx:02d}",
                    image_path=image_png,
                    mask_path=mask_png,
                    mask_index=idx,
                )
            )
    cases.sort(key=lambda c: (c.image_path.parent.name, c.mask_index))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


# ---------------------------------------------------------------------------
# Image / mask loading. Returns RGBA uint8 numpy. Real pipeline takes mx.array
# but the stub is fine with numpy and the boundary conversion lives there.
# ---------------------------------------------------------------------------


def load_rgba(image_path: Path, mask_path: Path) -> np.ndarray:
    """Load RGB image + grayscale mask -> RGBA uint8 (H, W, 4).

    Falls back to a 64x64 random RGBA if PIL is unavailable so the harness
    can still smoke-test on systems missing image deps.
    """
    try:
        from PIL import Image  # local import; eval doesn't require PIL
    except ImportError:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(64, 64, 4), dtype=np.uint8)

    img = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape[:2] != img.shape[:2]:
        # Resize mask to image (nearest) if shapes differ.
        mask_img = Image.open(mask_path).convert("L").resize(
            (img.shape[1], img.shape[0]), Image.NEAREST
        )
        mask = np.asarray(mask_img)
    rgba = np.concatenate([img, mask[..., None]], axis=-1).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Per-case runner.
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    case: str
    image_path: str
    mask_path: str
    seed: int
    status: str
    error: str = ""
    total_sec: float = 0.0
    ss_sec: float = 0.0
    slat_sec: float = 0.0
    decode_sec: float = 0.0
    n_gaussians: int = 0
    n_voxels: int = 0
    ply_bytes: int = 0
    out_ply: str = ""


def run_case(
    pipeline: Any,
    case: Case,
    seed: int,
    out_dir: Path,
    loader: Callable[[Path, Path], np.ndarray] = load_rgba,
) -> RowResult:
    case_dir = out_dir / f"case_{case.name}_seed_{seed}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ply_path = case_dir / "splat.ply"

    row = RowResult(
        case=case.name,
        image_path=str(case.image_path),
        mask_path=str(case.mask_path),
        seed=seed,
        status="fail",
        out_ply=str(ply_path),
    )

    try:
        image = loader(case.image_path, case.mask_path)
    except Exception as exc:  # pragma: no cover - I/O failures
        row.error = f"load_rgba: {exc}"
        return row

    t0 = time.perf_counter()
    try:
        out = pipeline(image, seed=seed, case_name=case.name)
    except TypeError:
        # Real pipeline likely won't take ``case_name`` kwarg.
        out = pipeline(image, seed=seed)
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        row.total_sec = time.perf_counter() - t0
        return row
    row.total_sec = time.perf_counter() - t0

    timings = out.get("timings", {}) or {}
    row.ss_sec = float(timings.get("ss_sec", 0.0))
    row.slat_sec = float(timings.get("slat_sec", 0.0))
    row.decode_sec = float(timings.get("decode_sec", 0.0))

    voxels = out.get("voxels")
    row.n_voxels = 0 if voxels is None else int(np.asarray(voxels).shape[0])
    gs = out.get("gs_params")
    if isinstance(gs, dict) and "_xyz" in gs:
        row.n_gaussians = int(np.asarray(gs["_xyz"]).shape[0])

    ply_bytes = out.get("ply_bytes")
    if ply_bytes is None:
        # Fallback: try real save_gaussian_ply if gs_params look complete.
        try:
            from meadow_wb.models.decoder_mlx import save_gaussian_ply  # type: ignore

            save_gaussian_ply(gs, str(ply_path))
            ply_bytes_count = ply_path.stat().st_size
        except Exception as exc:  # pragma: no cover
            row.error = f"ply_save: {exc}"
            return row
    else:
        ply_path.write_bytes(ply_bytes)
        ply_bytes_count = len(ply_bytes)
    row.ply_bytes = int(ply_bytes_count)

    row.status = "ok"
    return row


# ---------------------------------------------------------------------------
# Aggregation + reporting.
# ---------------------------------------------------------------------------


def aggregate(rows: List[RowResult]) -> Dict[str, Any]:
    ok = [r for r in rows if r.status == "ok"]
    fail = [r for r in rows if r.status != "ok"]

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    seed_groups: Dict[int, List[RowResult]] = {}
    for r in ok:
        seed_groups.setdefault(r.seed, []).append(r)

    seed_compare = {}
    for s, items in seed_groups.items():
        seed_compare[s] = {
            "n_cases": len(items),
            "ply_bytes_mean": float(np.mean([r.ply_bytes for r in items])) if items else 0.0,
            "n_gaussians_mean": float(np.mean([r.n_gaussians for r in items])) if items else 0.0,
        }

    return {
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "success_rate": (len(ok) / len(rows)) if rows else 0.0,
        "total_sec": _stats([r.total_sec for r in ok]),
        "ss_sec": _stats([r.ss_sec for r in ok]),
        "slat_sec": _stats([r.slat_sec for r in ok]),
        "decode_sec": _stats([r.decode_sec for r in ok]),
        "n_gaussians": _stats([r.n_gaussians for r in ok]),
        "ply_bytes": _stats([r.ply_bytes for r in ok]),
        "seed_compare": seed_compare,
        "failures": [{"case": r.case, "seed": r.seed, "error": r.error} for r in fail],
    }


def write_csv(rows: List[RowResult], path: Path) -> None:
    fields = [
        "case",
        "image_path",
        "mask_path",
        "seed",
        "status",
        "error",
        "total_sec",
        "ss_sec",
        "slat_sec",
        "decode_sec",
        "n_gaussians",
        "n_voxels",
        "ply_bytes",
        "out_ply",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def write_json(rows: List[RowResult], summary: Dict[str, Any], path: Path) -> None:
    payload = {
        "rows": [asdict(r) for r in rows],
        "summary": summary,
    }
    path.write_text(json.dumps(payload, indent=2))


def print_table(rows: List[RowResult], summary: Dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"  Eval summary  ({summary['n_ok']}/{summary['n_total']} ok, "
          f"success={summary['success_rate']*100:.1f}%)")
    print("=" * 78)

    def _fmt(stats: Dict[str, float]) -> str:
        return (f"mean={stats['mean']:.4f}  p50={stats['p50']:.4f}  "
                f"p95={stats['p95']:.4f}")

    print(f"  total_sec    : {_fmt(summary['total_sec'])}")
    print(f"  ss_sec       : {_fmt(summary['ss_sec'])}")
    print(f"  slat_sec     : {_fmt(summary['slat_sec'])}")
    print(f"  decode_sec   : {_fmt(summary['decode_sec'])}")
    print(f"  n_gaussians  : {_fmt(summary['n_gaussians'])}")
    print(f"  ply_bytes    : {_fmt(summary['ply_bytes'])}")

    if summary["seed_compare"]:
        print("\n  Per-seed compare:")
        for s, info in sorted(summary["seed_compare"].items()):
            print(f"    seed={s}  n={info['n_cases']}  "
                  f"mean_bytes={info['ply_bytes_mean']:.0f}  "
                  f"mean_gauss={info['n_gaussians_mean']:.0f}")

    if summary["failures"]:
        print("\n  Failures:")
        for f in summary["failures"]:
            print(f"    [{f['seed']}] {f['case']}: {f['error']}")
    print()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", type=Path, required=True,
                   help="Root dir to scan for image.png + numbered masks.")
    p.add_argument("--out-dir", type=Path, default=Path("meadow_wb/_eval_out"),
                   help="Where to write per-case dirs + eval_summary.{csv,json}.")
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated seed list, e.g. '42,43'.")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Cap discovered (image, mask) pairs (per-seed cases = this x len(seeds)).")
    p.add_argument("--pipeline", choices=("stub", "real"), default="stub",
                   help="'stub' uses mock; 'real' loads SAM3DObjectsPipeline.")
    p.add_argument("--npz-dir", type=str, default=None,
                   help="Weights dir for the real pipeline (ignored by stub).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-row print.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cases = discover_cases(args.image_dir, max_cases=args.max_cases)
    if not cases:
        print(f"[eval] no cases found under {args.image_dir}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"[eval] {len(cases)} cases x {len(seeds)} seeds = "
              f"{len(cases) * len(seeds)} runs")

    pipeline = load_pipeline(args.pipeline, args.npz_dir)

    rows: List[RowResult] = []
    for case in cases:
        for seed in seeds:
            row = run_case(pipeline, case, seed, args.out_dir)
            rows.append(row)
            if not args.quiet:
                tag = "OK " if row.status == "ok" else "FAIL"
                print(f"  [{tag}] {case.name} seed={seed} "
                      f"t={row.total_sec:.2f}s gauss={row.n_gaussians} "
                      f"bytes={row.ply_bytes}"
                      + (f"  err={row.error}" if row.error else ""))

    summary = aggregate(rows)
    write_csv(rows, args.out_dir / "eval_summary.csv")
    write_json(rows, summary, args.out_dir / "eval_summary.json")
    print_table(rows, summary)
    return 0 if summary["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
