"""MLX port of SAM 3D Objects ``latent_share_transformer`` (ss_flow MOT).

Mirrors PT ``SparseStructureFlowTdfyWrapper.merge_latent_share_transformer``
and ``split_latent_share_transformer`` in
``sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py``.

CRITICAL FINDING — this is **NOT** an attention-based transformer.
=================================================================
Despite the name, ``latent_share_transformer`` has **zero learned parameters**.
It is a pure structural reshape:

* PT source ``merge_latent_share_transformer`` is a single ``torch.cat`` along
  the token dimension (``dim=1``).
* PT source ``split_latent_share_transformer`` is the inverse: slice the
  concatenated tensor back into its parts using each per-modality
  ``pos_emb.shape[0]`` as the stride.
* ``ss_flow.npz`` confirms it: there are **0 keys** with prefix
  ``latent_share_transformer.``  All learned parameters live in
  ``latent_mapping.{modality}.*`` and are loaded by ``LatentMapping`` /
  ``OutputMapping``.

So the only thing this module does is enforce the merge mapping (config) and
provide ``forward(latents) -> merged`` and ``inverse(merged) -> latents``.

The ss_flow MOT canonical config (derived from the npz weights — DiT blocks
were trained with ``latent_names = ["shape", "6drotation_normalized"]``)::

    latent_share_transformer = {
        "6drotation_normalized": [
            "6drotation_normalized",
            "translation",
            "scale",
            "translation_scale",
        ],
    }

That is, ``shape`` (token_len 4096) is a passthrough; the four small-token
modalities (each token_len 1) are concatenated along the token axis under
the merged key ``6drotation_normalized`` (total token_len 4).  The DiT then
operates on two streams: ``shape`` (4096 tokens) and ``6drotation_normalized``
(4 tokens).

Usage in an inference pipeline::

    # After LatentMapping.project_dict has produced per-modality (B, N_i, 1024):
    merger = LatentShareTransformer(SS_FLOW_MERGE_MAP)
    dit_input = merger(per_modality)        # 5 -> 2 streams
    dit_output = dit_backbone(dit_input)    # 2 -> 2 streams
    splitter = LatentSplitTransformer(SS_FLOW_MERGE_MAP, token_lens)
    per_modality_out = splitter(dit_output) # 2 -> 5 streams
    # Then OutputMapping.project_dict for per-modality velocity.

Strict MLX rules: pure MLX, no numpy/torch in the hot path.  Because there
are no learned weights, only ``mx.concatenate`` and slicing are required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Canonical ss_flow merge map
# ---------------------------------------------------------------------------


SS_FLOW_MERGE_MAP: Dict[str, List[str]] = {
    "6drotation_normalized": [
        # Order MUST match PT ss_generator.yaml latent_share_transformer
        # (translation BEFORE scale).
        "6drotation_normalized",
        "translation",
        "scale",
        "translation_scale",
    ],
}
"""Default merge config for ss_flow MOT.

Derived from the ``ss_flow.npz`` weights: the DiT blocks (``blocks.{i}.norm2.*``,
``blocks.{i}.mlp.*`` etc.) only have per-modality keys for ``shape`` and
``6drotation_normalized`` — so those are the two stream names the DiT was
trained with.  ``shape`` is not in the merge map (it passes through unchanged);
the other four modalities are concatenated along the token dim under the
``6drotation_normalized`` bucket.
"""


SS_FLOW_TOKEN_LENS: Dict[str, int] = {
    "shape": 4096,
    "6drotation_normalized": 1,
    "scale": 1,
    "translation": 1,
    "translation_scale": 1,
}
"""Per-modality token_len for ss_flow MOT.  ``shape`` uses a 16^3 voxel APE
table; the other four are single-token modalities.  These values are a
property of the data + ``LatentMapping.pos_emb`` shapes, *not* of
``latent_share_transformer`` itself — included here so callers that don't
have a constructed ``LatentMapping`` handy can still drive the splitter."""


# ---------------------------------------------------------------------------
# Forward merge
# ---------------------------------------------------------------------------


class LatentShareTransformer(nn.Module):
    """Concatenate-along-token merge of multiple modality latents.

    No learned parameters.  Mirrors PT ``merge_latent_share_transformer``::

        for merged_name, latent_names in self.latent_share_transformer.items():
            tensors = [latent_dict[n] for n in latent_names]
            return_dict[merged_name] = torch.cat(tensors, dim=1)
        for n in latent_dict:
            if n not in visited:
                return_dict[n] = latent_dict[n]   # passthrough
    """

    def __init__(self, merge_map: Optional[Mapping[str, List[str]]] = None):
        super().__init__()
        if merge_map is None:
            merge_map = SS_FLOW_MERGE_MAP
        # Defensive copy — also normalises lists.
        self.merge_map: Dict[str, List[str]] = {
            k: list(v) for k, v in merge_map.items()
        }
        # Pre-compute the set of modalities that get absorbed into a merged
        # group, so passthrough is O(1) per key.
        self._merged_inputs = {
            n for names in self.merge_map.values() for n in names
        }

    # ---- forward -------------------------------------------------------
    def __call__(self, latents: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}
        # 1) Build merged groups in a deterministic order (Python 3.7+ dict
        #    insertion-order).
        for merged_name, latent_names in self.merge_map.items():
            tensors = []
            for n in latent_names:
                if n not in latents:
                    raise KeyError(
                        f"LatentShareTransformer: missing input modality "
                        f"'{n}' for merged group '{merged_name}'. "
                        f"Available: {sorted(latents.keys())}"
                    )
                tensors.append(latents[n])
            out[merged_name] = mx.concatenate(tensors, axis=1)

        # 2) Passthrough every other modality untouched.
        for n, t in latents.items():
            if n not in self._merged_inputs and n not in out:
                out[n] = t
        return out

    forward = __call__  # PT-style alias

    # ---- npz construction ---------------------------------------------
    @classmethod
    def from_npz(
        cls,
        weights_dict: Any = None,  # noqa: ARG003 — accepted for API symmetry
        prefix: str = "reverse_fn.backbone.latent_share_transformer.",  # noqa: ARG003
        merge_map: Optional[Mapping[str, List[str]]] = None,
    ) -> "LatentShareTransformer":
        """Construct from npz weights.

        ``latent_share_transformer`` has **no** learned parameters in the
        ss_flow checkpoint — verified empirically: 0 keys under
        ``reverse_fn.backbone.latent_share_transformer.*``.  This factory
        therefore ignores ``weights_dict`` / ``prefix`` and just instantiates
        the canonical merge map (or a caller-supplied one).

        The signature accepts ``weights_dict`` and ``prefix`` only so that the
        caller code can be uniform with the other ``from_npz`` factories
        (``LatentMapping.from_npz`` etc.).
        """
        return cls(merge_map=merge_map)


# ---------------------------------------------------------------------------
# Inverse split
# ---------------------------------------------------------------------------


class LatentSplitTransformer(nn.Module):
    """Inverse of :class:`LatentShareTransformer`.

    Mirrors PT ``split_latent_share_transformer``::

        for merged_name, latent_names in self.latent_share_transformer.items():
            start = 0
            tensors = output_latents[merged_name]
            for n in latent_names:
                token_len = self.latent_mapping[n].pos_emb.shape[0]
                return_dict[n] = tensors[:, start : start + token_len]
                start += token_len
        for n in output_latents:
            if n not in visited:
                return_dict[n] = output_latents[n]

    Token lengths must be supplied (typically read from
    ``LatentMapping.modalities[n].pos_emb.shape[0]``).
    """

    def __init__(
        self,
        merge_map: Optional[Mapping[str, List[str]]] = None,
        token_lens: Optional[Mapping[str, int]] = None,
    ):
        super().__init__()
        if merge_map is None:
            merge_map = SS_FLOW_MERGE_MAP
        if token_lens is None:
            token_lens = SS_FLOW_TOKEN_LENS
        self.merge_map: Dict[str, List[str]] = {
            k: list(v) for k, v in merge_map.items()
        }
        # Validate every merged-input has a token_len entry.
        for merged_name, names in self.merge_map.items():
            for n in names:
                if n not in token_lens:
                    raise ValueError(
                        f"LatentSplitTransformer: token_lens missing entry for "
                        f"'{n}' (needed by merged group '{merged_name}')."
                    )
        self.token_lens: Dict[str, int] = dict(token_lens)

    # ---- forward -------------------------------------------------------
    def __call__(self, merged: Dict[str, mx.array]) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}
        # 1) Slice merged groups back into per-modality tensors.
        for merged_name, latent_names in self.merge_map.items():
            if merged_name not in merged:
                raise KeyError(
                    f"LatentSplitTransformer: missing merged stream "
                    f"'{merged_name}'. Available: {sorted(merged.keys())}"
                )
            tensor = merged[merged_name]
            start = 0
            for n in latent_names:
                tl = self.token_lens[n]
                # mx slice along axis=1
                out[n] = tensor[:, start : start + tl]
                start += tl
            # Sanity: total tokens consumed must match input.
            if start != tensor.shape[1]:
                raise ValueError(
                    f"LatentSplitTransformer: merged group '{merged_name}' "
                    f"has {tensor.shape[1]} tokens but token_lens sum to "
                    f"{start} for {latent_names}."
                )

        # 2) Passthrough every other (un-merged) modality.
        merged_keys = set(self.merge_map.keys())
        for n, t in merged.items():
            if n not in merged_keys:
                out[n] = t
        return out

    forward = __call__  # PT-style alias

    # ---- npz construction ---------------------------------------------
    @classmethod
    def from_npz(
        cls,
        weights_dict: Any = None,  # noqa: ARG003
        prefix: str = "reverse_fn.backbone.latent_share_transformer.",  # noqa: ARG003
        merge_map: Optional[Mapping[str, List[str]]] = None,
        token_lens: Optional[Mapping[str, int]] = None,
    ) -> "LatentSplitTransformer":
        """No learned weights — see :meth:`LatentShareTransformer.from_npz`.

        For convenience, callers can omit ``token_lens`` (defaults to
        :data:`SS_FLOW_TOKEN_LENS`) or pass them in directly.  When the
        per-modality ``LatentMapping`` is already constructed, prefer::

            token_lens = {n: lm.modalities[n].token_len for n in lm.modality_names}
            split = LatentSplitTransformer(merge_map, token_lens)
        """
        return cls(merge_map=merge_map, token_lens=token_lens)
