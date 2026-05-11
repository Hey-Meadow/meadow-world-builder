"""NPZ weight loader for MLX modules.

Used by every model file in ``meadow_sb/models/`` to consume the npz dumps
produced by Agent G's ``scripts/convert_weights.py``.

Two entry points:

* :func:`load_npz_module` — read an npz file from disk and return a flat dict
  ``{key: mx.array}``. Optionally strip a leading ``module_prefix.`` so callers
  can target a single sub-module.

* :func:`attach_weights` — take that flat dict and bulk-assign into an MLX
  ``nn.Module`` via ``tree_unflatten`` + ``module.update``. Supports a
  ``key_remap`` dict for the (frequent) case where PT and MLX param names
  diverge (e.g. ``mlp.0.weight`` -> ``fc1.weight``).

Both helpers are dtype-agnostic; the caller decides float32 vs bf16 via
explicit ``.astype`` post-load.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten


# ---------------------------------------------------------------------------
# load_npz_module
# ---------------------------------------------------------------------------


def load_npz_module(
    path: str,
    module_prefix: Optional[str] = None,
    dtype: Optional[mx.Dtype] = None,
) -> Dict[str, mx.array]:
    """Load an npz file and return ``{key: mx.array}``.

    Parameters
    ----------
    path:
        Path to a ``.npz`` produced by ``np.savez(_compressed)``.
    module_prefix:
        Optional dotted prefix to strip. ``"encoder"`` strips ``encoder.`` from
        the front of each key. Keys without the prefix are dropped.
    dtype:
        If set, every array is cast to this MLX dtype. Default keeps the npz
        dtype (typically float32).
    """
    with np.load(path, allow_pickle=False) as zf:
        items = {k: zf[k] for k in zf.files}

    out: Dict[str, mx.array] = {}
    prefix = (module_prefix.rstrip(".") + ".") if module_prefix else None
    for k, v in items.items():
        if prefix is not None:
            if not k.startswith(prefix):
                continue
            k = k[len(prefix):]
        arr = mx.array(v)
        if dtype is not None:
            arr = arr.astype(dtype)
        out[k] = arr
    return out


# ---------------------------------------------------------------------------
# attach_weights
# ---------------------------------------------------------------------------


def _apply_remap(
    weights: Mapping[str, mx.array],
    key_remap: Optional[Mapping[str, str] | Callable[[str], Optional[str]]],
) -> Dict[str, mx.array]:
    """Rename keys per ``key_remap``.

    * dict form: literal ``old -> new`` replacements (whole-key match).
      Keys not in the dict pass through unchanged.
    * callable form: ``fn(key) -> new_key or None``; ``None`` drops the key.
    """
    if key_remap is None:
        return dict(weights)
    if callable(key_remap):
        out: Dict[str, mx.array] = {}
        for k, v in weights.items():
            new_k = key_remap(k)
            if new_k is not None:
                out[new_k] = v
        return out
    # dict
    out2: Dict[str, mx.array] = {}
    for k, v in weights.items():
        out2[key_remap.get(k, k)] = v
    return out2


def attach_weights(
    mlx_module: nn.Module,
    npz_dict: Mapping[str, mx.array],
    key_remap: Optional[Mapping[str, str] | Callable[[str], Optional[str]]] = None,
    strict: bool = False,
) -> Dict[str, list]:
    """Bulk-assign ``npz_dict`` into ``mlx_module`` via ``tree_unflatten``.

    Parameters
    ----------
    mlx_module:
        Target module. Must already be instantiated (so its parameter tree is
        defined). MLX's ``Module.update`` only writes leaves that already exist,
        so missing keys are tolerated by MLX itself.
    npz_dict:
        Flat ``{dotted_key: mx.array}`` dict (e.g. from :func:`load_npz_module`).
    key_remap:
        Optional rename map or callable. See :func:`_apply_remap`.
    strict:
        If ``True``, raise on any key in ``npz_dict`` that wasn't consumed
        (after remap) by ``mlx_module``, or any module param that wasn't
        present in ``npz_dict``.

    Returns
    -------
    diag:
        ``{"missing": [...], "unexpected": [...]}`` for caller-side logging.
    """
    remapped = _apply_remap(npz_dict, key_remap)

    # Build the nested tree MLX expects and call update.
    nested = tree_unflatten(list(remapped.items()))
    mlx_module.update(nested)

    # Diagnostics: compare leaves of module.parameters() to remapped keys.
    have_keys = set(k for k, _ in tree_flatten(mlx_module.parameters()))
    given_keys = set(remapped.keys())
    missing = sorted(have_keys - given_keys)
    unexpected = sorted(given_keys - have_keys)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"attach_weights strict mode: "
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    return {"missing": missing, "unexpected": unexpected}


__all__ = ["load_npz_module", "attach_weights"]
