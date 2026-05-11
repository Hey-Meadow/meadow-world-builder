"""MLX port of SAM 3D Objects per-modality latent mapping.

Mirrors PT class ``Latent`` in
``sam3d_objects/model/backbone/tdfy_dit/models/mm_latent.py`` and the
``SparseStructureFlowTdfyWrapper.latent_mapping`` ModuleDict in
``mot_sparse_structure_flow.py``.

This module sits BETWEEN the EmbedderFuser (image conditioning) and the
DiT backbone:

    raw latent (B, N, in_dim)
        -> input_layer:  Linear(in_dim, model_channels=1024)
        -> + pos_emb[None]                                          # APE
        ==> DiT input  (B, N, 1024)

After DiT:

    DiT output (B, N, 1024)
        -> layer_norm(last_dim)                                     # no params
        -> out_layer:    Linear(model_channels, in_dim)
        ==> velocity in latent space (B, N, in_dim)

Two flavors observed in the npz weights:

* ``ss_flow.npz`` (MOT, multi-modality):
    keys = ``reverse_fn.backbone.latent_mapping.{modality}.{input_layer|out_layer}.{weight|bias}``
           ``reverse_fn.backbone.latent_mapping.{modality}.pos_emb``
    modalities = ``shape, 6drotation_normalized, scale, translation, translation_scale``
    in_dim     = ``8, 6, 3, 3, 1`` respectively
    pos_emb shape:
        - ``shape`` -> (4096, 1024) — sin/cos APE over a 16^3 voxel grid
                                       (``ShapePositionEmbedder``, fixed buffer)
        - others   -> (1, 1024)    — learnt single-token APE
                                       (``LearntPositionEmbedder``)

* ``slat_flow.npz`` (single-modality, sparse):
    keys = ``reverse_fn.backbone.{input_layer|out_layer}.{weight|bias}``
    in_dim = 8, model_channels = 128 (sparse pre/post layers, NOT 1024)
    NOTE: this is the sparse path; full sparse 3D handling is OBJ-METAL-SPARSE.
    The dense ``LatentMapping``/``OutputMapping`` interface here covers the
    weight load + linear projection only.

Strict MLX rules: no numpy/torch in the inference hot path; nn.Linear, nn.LayerNorm,
``mx.fast.layer_norm`` only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Per-modality submodule
# ---------------------------------------------------------------------------


class _LatentPerModality(nn.Module):
    """Per-modality input/output projection + (frozen or learnt) pos_emb.

    Mirrors PT ``Latent`` in ``mm_latent.py``.  Holds:
        - ``input_layer``  : Linear(in_dim -> model_channels)
        - ``out_layer``    : Linear(model_channels -> in_dim)
        - ``pos_emb``      : (token_len, model_channels) buffer/parameter
    """

    def __init__(self, in_dim: int, model_channels: int, token_len: int):
        super().__init__()
        self.in_dim = in_dim
        self.model_channels = model_channels
        self.token_len = token_len

        self.input_layer = nn.Linear(in_dim, model_channels, bias=True)
        self.out_layer = nn.Linear(model_channels, in_dim, bias=True)
        # Treated as a frozen parameter for inference; PT distinguishes
        # buffer (sin/cos) vs Parameter (learnt) but for a forward pass they
        # behave identically — both are added to ``input_layer(x)``.
        self.pos_emb = mx.zeros((token_len, model_channels))

    def to_input(self, x: mx.array) -> mx.array:
        """latent (B, N, in_dim) -> projected + pos_emb-added (B, N, model_channels)."""
        x = self.input_layer(x)
        # pos_emb (N, C) broadcasts over batch via [None]
        return x + self.pos_emb[None]

    def to_output(self, h: mx.array) -> mx.array:
        """DiT output (B, N, model_channels) -> velocity (B, N, in_dim).

        PT applies ``F.layer_norm(h, h.shape[-1:])`` (i.e. last-dim normalize,
        no learnable affine), then the output Linear.
        """
        # Use mx.fast.layer_norm with no weight/bias (affine=False equivalent).
        # mx.fast.layer_norm(x, weight, bias, eps) — pass None weight/bias.
        h = mx.fast.layer_norm(h, None, None, 1e-5)
        return self.out_layer(h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LatentMapping(nn.Module):
    """Per-modality input projection container (multi- or single-modality).

    For MOT (ss_flow) this acts as a ModuleDict keyed by modality name.
    For single-modality (slat_flow) it carries a single ``input_layer`` /
    ``out_layer`` pair under the empty-key entry.

    Forward:
        ``__call__(x, modality)`` returns ``input_layer(x) + pos_emb[None]``.
    """

    def __init__(
        self,
        modalities: Dict[str, "_LatentMappingSpec"],
        model_channels: int = 1024,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.modalities: Dict[str, _LatentPerModality] = {
            name: _LatentPerModality(spec.in_dim, model_channels, spec.token_len)
            for name, spec in modalities.items()
        }
        # Keep ordering stable (insertion order preserved in py3.7+).
        self.modality_names: List[str] = list(modalities.keys())

    # ---- forward ------------------------------------------------------
    def __call__(self, x: mx.array, modality: Optional[str] = None) -> mx.array:
        if modality is None:
            if len(self.modality_names) != 1:
                raise ValueError(
                    f"`modality` is required for multi-modality LatentMapping; "
                    f"got names={self.modality_names}"
                )
            modality = self.modality_names[0]
        return self.modalities[modality].to_input(x)

    def to_input(self, x: mx.array, modality: Optional[str] = None) -> mx.array:
        return self(x, modality)

    def project_dict(self, latents: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Per-modality forward over a dict of latents (matches PT
        ``project_input`` minus the ``latent_share_transformer`` merge)."""
        return {n: self.modalities[n].to_input(latents[n]) for n in latents}

    # ---- npz loading --------------------------------------------------
    @classmethod
    def from_npz(
        cls,
        npz_path_or_weights,
        prefix: str = "reverse_fn.backbone.latent_mapping.",
        model_channels: int = 1024,
    ) -> "LatentMapping":
        """Construct from npz weights.

        ``npz_path_or_weights`` may be a path string or an already-loaded dict
        of ``mx.array`` (so callers can share ``mx.load`` between LatentMapping
        and OutputMapping without re-reading the file).

        Two prefix conventions are supported:

        1. MOT  : ``{prefix}{modality}.{input_layer|out_layer}.{weight|bias}``
                  with ``{prefix}{modality}.pos_emb``.
                  (default ``prefix='reverse_fn.backbone.latent_mapping.'``).

        2. Single-modality (slat_flow): set ``prefix='reverse_fn.backbone.'`` —
                  keys are ``{prefix}input_layer.weight`` etc.  No pos_emb.
        """
        sd = _load_weights(npz_path_or_weights)
        modalities = _discover_modalities(sd, prefix)
        model = cls(modalities, model_channels=model_channels)
        _load_per_modality(model.modalities, sd, prefix, modalities)
        return model


class OutputMapping(nn.Module):
    """Per-modality output projection container.

    Mirrors PT ``project_output``: applies ``F.layer_norm`` (no affine) then
    per-modality ``out_layer``.  Shares the same parameters as ``LatentMapping``
    because PT stores ``input_layer`` and ``out_layer`` together under the same
    ``latent_mapping.{modality}`` namespace — so ``OutputMapping.from_npz``
    loads the ``out_layer`` half of those keys.
    """

    def __init__(
        self,
        modalities: Dict[str, "_LatentMappingSpec"],
        model_channels: int = 1024,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.modalities: Dict[str, _LatentPerModality] = {
            name: _LatentPerModality(spec.in_dim, model_channels, spec.token_len)
            for name, spec in modalities.items()
        }
        self.modality_names: List[str] = list(modalities.keys())

    def __call__(self, x: mx.array, modality: Optional[str] = None) -> mx.array:
        if modality is None:
            if len(self.modality_names) != 1:
                raise ValueError(
                    f"`modality` is required for multi-modality OutputMapping; "
                    f"got names={self.modality_names}"
                )
            modality = self.modality_names[0]
        return self.modalities[modality].to_output(x)

    def to_output(self, x: mx.array, modality: Optional[str] = None) -> mx.array:
        return self(x, modality)

    def project_dict(self, latents: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return {n: self.modalities[n].to_output(latents[n]) for n in latents}

    @classmethod
    def from_npz(
        cls,
        npz_path_or_weights,
        prefix: str = "reverse_fn.backbone.latent_mapping.",
        model_channels: int = 1024,
    ) -> "OutputMapping":
        sd = _load_weights(npz_path_or_weights)
        modalities = _discover_modalities(sd, prefix)
        model = cls(modalities, model_channels=model_channels)
        _load_per_modality(model.modalities, sd, prefix, modalities)
        return model


class PositionalEmbedding(nn.Module):
    """Standalone wrapper around a learned (or fixed sin/cos) APE table.

    Provided for callers that want to manage pos_emb separately from the
    input/output projections.  In the ss_flow / slat_flow weights this is
    redundant (pos_emb lives inside ``latent_mapping.{modality}``), but the
    spec asks for it as a top-level class.

    Forward: ``__call__(x)`` returns ``x + pos_emb[None]`` (token-broadcast).
    Optional ``positions`` lets the caller index a subset of the table.
    """

    def __init__(self, token_len: int, model_channels: int):
        super().__init__()
        self.token_len = token_len
        self.model_channels = model_channels
        self.pos_emb = mx.zeros((token_len, model_channels))

    def __call__(
        self,
        x: mx.array,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        if positions is None:
            return x + self.pos_emb[None]
        # Gather rows from the table.  positions: (B, N) or (N,) of int indices.
        sel = self.pos_emb[positions]  # broadcast-friendly fancy index
        if sel.ndim == 2:
            sel = sel[None]
        return x + sel

    @classmethod
    def from_npz(
        cls,
        npz_path_or_weights,
        key: str,
    ) -> "PositionalEmbedding":
        """Load a single pos_emb table by exact key (e.g.
        ``'reverse_fn.backbone.latent_mapping.shape.pos_emb'``)."""
        sd = _load_weights(npz_path_or_weights)
        if key not in sd:
            raise KeyError(f"pos_emb key not found: {key}")
        arr = sd[key]
        token_len, model_channels = arr.shape
        m = cls(token_len, model_channels)
        m.pos_emb = arr
        return m


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _LatentMappingSpec:
    """Per-modality dimensions discovered from npz."""

    __slots__ = ("in_dim", "token_len")

    def __init__(self, in_dim: int, token_len: int):
        self.in_dim = in_dim
        self.token_len = token_len


def _load_weights(npz_path_or_weights):
    if isinstance(npz_path_or_weights, str):
        return mx.load(npz_path_or_weights)
    return npz_path_or_weights


def _discover_modalities(sd, prefix: str) -> Dict[str, _LatentMappingSpec]:
    """Walk weights to find modality names and their dims.

    Two cases:
    1. MOT prefix ends in ``latent_mapping.`` -> modality name is the next
       component before ``.input_layer`` / ``.pos_emb``.
    2. Direct prefix (slat_flow) ends in ``backbone.`` and there's no
       per-modality nesting.  We treat this as a single anonymous modality
       keyed by ``""`` (empty string).
    """
    matches: Dict[str, _LatentMappingSpec] = {}
    if prefix.endswith("latent_mapping."):
        # MOT case
        seen: Dict[str, Dict[str, int]] = {}
        for k in sd.keys():
            if not k.startswith(prefix):
                continue
            tail = k[len(prefix) :]
            parts = tail.split(".")
            if len(parts) < 2:
                continue
            modality = parts[0]
            seen.setdefault(modality, {})
            if parts[1] == "input_layer" and parts[-1] == "weight":
                seen[modality]["in_dim"] = sd[k].shape[1]
            elif parts[1] == "pos_emb":
                seen[modality]["token_len"] = sd[k].shape[0]
        for modality, dims in seen.items():
            if "in_dim" not in dims:
                continue
            token_len = dims.get("token_len", 1)
            matches[modality] = _LatentMappingSpec(dims["in_dim"], token_len)
    else:
        # Single-modality case: look for direct input_layer.weight under prefix
        in_dim = None
        for k in sd.keys():
            if k == prefix + "input_layer.weight":
                in_dim = sd[k].shape[1]
                break
        if in_dim is None:
            raise ValueError(
                f"No input_layer.weight under prefix={prefix!r}; "
                f"available keys (first 5): {list(sd.keys())[:5]}"
            )
        # No pos_emb in single-modality slat layout
        matches[""] = _LatentMappingSpec(in_dim, token_len=1)
    return matches


def _load_per_modality(
    target: Dict[str, _LatentPerModality],
    sd,
    prefix: str,
    specs: Dict[str, _LatentMappingSpec],
) -> None:
    """Copy weights into the constructed per-modality submodules."""
    for modality, _ in specs.items():
        m = target[modality]
        if modality == "":
            base = prefix  # ends in "."
        else:
            base = f"{prefix}{modality}."
        m.input_layer.weight = sd[base + "input_layer.weight"]
        m.input_layer.bias = sd[base + "input_layer.bias"]
        m.out_layer.weight = sd[base + "out_layer.weight"]
        m.out_layer.bias = sd[base + "out_layer.bias"]
        pos_key = base + "pos_emb"
        if pos_key in sd:
            m.pos_emb = sd[pos_key]
        # else leave as zeros (single-modality slat has no pos_emb at this stage)
