"""MLX <-> NumPy / PyTorch interop helpers.

Shared by every agent in the YoNoSplat MLX port. Keep this file dependency-light
(no model imports) so tests in any module can pull these without circular imports.

Convention:
- We accept ``torch.Tensor``, ``np.ndarray``, ``mx.array``, or Python scalars in
  ``pt_to_mlx`` and always return ``mx.array``.
- ``mlx_to_np`` always returns a contiguous ``np.ndarray``.
- ``assert_close`` is the canonical numeric-equivalence helper for cross-impl
  tests; it prints the diagnostic on failure rather than swallowing it.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------


def _np_to_mx_dtype(np_dtype: np.dtype):
    """Map a numpy dtype to the closest MLX dtype.

    MLX does not have float64 on Metal; we downcast to float32 (with a soft
    warning suppressed — caller is responsible for the precision contract).
    int64 likewise gets downcast to int32.
    """
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.float32:
        return mx.float32
    if np_dtype == np.float64:
        return mx.float32
    if np_dtype == np.float16:
        return mx.float16
    if np_dtype == np.int64:
        return mx.int32
    if np_dtype == np.int32:
        return mx.int32
    if np_dtype == np.int16:
        return mx.int16
    if np_dtype == np.int8:
        return mx.int8
    if np_dtype == np.uint8:
        return mx.uint8
    if np_dtype == np.bool_:
        return mx.bool_
    # default: let MLX pick from raw bytes
    return None


# ---------------------------------------------------------------------------
# pt_to_mlx
# ---------------------------------------------------------------------------


def pt_to_mlx(x: Any) -> mx.array:
    """Convert torch.Tensor / np.ndarray / mx.array / python scalar to mx.array.

    Detached, on-CPU view; preserves shape; maps dtype via :func:`_np_to_mx_dtype`.
    bfloat16 torch tensors are upcast to float32 before crossing the boundary
    (numpy has no native bfloat16; MLX 0.31 ``mx.array`` constructor only
    accepts numpy / list / scalar inputs).
    """
    if isinstance(x, mx.array):
        return x

    # torch.Tensor — duck-typed to avoid import cost when torch not used
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        t = x.detach().cpu()
        # bf16 cannot cross via numpy
        if str(t.dtype) == "torch.bfloat16":
            t = t.float()
        arr = t.numpy()
        return mx.array(arr, dtype=_np_to_mx_dtype(arr.dtype))

    if isinstance(x, np.ndarray):
        # MLX won't accept non-contiguous arrays in some 0.31 builds.
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return mx.array(x, dtype=_np_to_mx_dtype(x.dtype))

    if isinstance(x, (int, float, bool, list, tuple)):
        arr = np.asarray(x)
        return mx.array(arr, dtype=_np_to_mx_dtype(arr.dtype))

    raise TypeError(f"pt_to_mlx: unsupported input type {type(x)!r}")


# ---------------------------------------------------------------------------
# mlx_to_np
# ---------------------------------------------------------------------------


def mlx_to_np(arr: mx.array) -> np.ndarray:
    """Materialize an MLX array as a contiguous numpy array.

    Calls ``mx.eval`` first so lazy graphs don't surface as ``nan``/garbage.
    bfloat16 is upcast to float32 because numpy cannot represent it.
    """
    if not isinstance(arr, mx.array):
        # Be forgiving: allow round-trip through np.asarray for stray scalars.
        return np.asarray(arr)
    mx.eval(arr)
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float32)
    return np.array(arr, copy=True)


# ---------------------------------------------------------------------------
# assert_close
# ---------------------------------------------------------------------------


def assert_close(
    mlx_out: Any,
    pt_out: Any,
    atol: float = 1e-4,
    rtol: float = 0.0,
    name: str = "",
) -> float:
    """Compare an MLX output against a torch/numpy reference; assert max-abs-diff.

    Returns the observed ``max_abs_diff`` (useful for ratcheting tolerances down
    in tests). On failure raises ``AssertionError`` with a structured message.
    """
    a = mlx_to_np(mlx_out) if isinstance(mlx_out, mx.array) else np.asarray(mlx_out)
    if hasattr(pt_out, "detach"):
        b = pt_out.detach().cpu()
        if str(b.dtype) == "torch.bfloat16":
            b = b.float()
        b = b.numpy()
    else:
        b = np.asarray(pt_out)

    if a.shape != b.shape:
        raise AssertionError(
            f"[assert_close{(' ' + name) if name else ''}] shape mismatch: "
            f"mlx={a.shape} ref={b.shape}"
        )

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0

    tag = f" [{name}]" if name else ""
    if max_abs > atol + rtol * float(np.abs(b).max() if b.size else 0.0):
        raise AssertionError(
            f"assert_close{tag} FAILED: max_abs={max_abs:.3e} "
            f"mean_abs={mean_abs:.3e} atol={atol:.1e} rtol={rtol:.1e} "
            f"shape={a.shape}"
        )
    return max_abs


__all__ = ["pt_to_mlx", "mlx_to_np", "assert_close"]
