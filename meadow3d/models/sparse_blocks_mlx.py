"""MLX port of slat_flow ``input_blocks`` / ``out_blocks`` (sparse 3D conv stack).

These wrap the ``DiTBackbone`` at the input/output of the SLAT generator. PT
source: ``sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py``
(``SparseResBlock3d`` and the ``input_blocks`` / ``out_blocks`` lists inside
``SLatFlowModel``).

Architecture (read off ``slat_flow.npz``)
----------------------------------------
slat_flow uses ``io_block_channels=[128]``, ``model_channels=1024``,
``num_io_res_blocks=2``, ``patch_size=2``, ``use_skip_connection=True``. That
gives::

    input_blocks.0  : SparseResBlock3d(128 -> 128)              (no down/up)
    input_blocks.1  : SparseResBlock3d(128 -> 1024, downsample) (1024 chans
                                                                 after halve)
    out_blocks.0    : SparseResBlock3d(2048 -> 128, upsample)   (input is
                                                                 [h | skip]
                                                                 concat = 2*1024)
    out_blocks.1    : SparseResBlock3d(256 -> 128)              ([h | skip] = 2*128)

Each ``SparseResBlock3d`` consumes::

    norm1.{weight,bias}            LayerNorm32  affine=True   over channels axis
    conv1.conv.{weight,bias}       SubMConv3d
    emb_layers.1.{weight,bias}     Linear(emb_channels=1024 -> 2*out_channels)
    conv2.conv.{weight,bias}       SubMConv3d  (zero-init in PT; loaded from ckpt)
    skip_connection.{weight,bias}  Linear (only when channels != out_channels)
    [no norm2 weights -- LayerNorm32 with affine=False]
    [no down/up weights -- SparseDownsample/Upsample are parameter-free]

Forward (matches PT line for line)::

    h = updown(x)                              # x is unchanged feats; coords may shift
    h = norm1(h_feats)                         # LN over channels
    h = silu(h)
    h = conv1(h, coords_h)                     # SubMConv3d on (post-updown) coords
    scale, shift = chunk(emb_layers(emb), 2)
    h = layer_norm_no_affine(h) * (1+scale) + shift
    h = silu(h)
    h = conv2(h, coords_h)
    h = h + skip_connection(updown(x).feats)   # SparseLinear if channels mismatch

Submanifold neighbor-table caching
----------------------------------
``SparseSubmConv3d.__call__`` calls ``build_neighbor_table(coords, ...)`` which
caches by ``id(coords)``. Both ``conv1`` and ``conv2`` of every block hit the
same coords array (post-updown), so within one forward we build at most TWO
neighbor tables per stage (one for the un-downsampled coord set used by
``input_blocks.0`` / ``out_blocks.last``, one for the downsampled coord set
used by everything in between). Across the 50 ODE steps the *same* coords
arrays are threaded through, so the cache holds and we pay the build cost
exactly once per inference, not 50× per inference.

Weight layout
-------------
The npz converter (``meadow3d/weights/convert.py``) ships SubMConv3d weights as
``(C_out, KH, KW, C_in, KD)`` -- the result of ``transpose(0, 2, 3, 4, 1)`` on
the PT-side state-dict shape ``(C_out, KD, KH, KW, C_in)``. To install into
``SparseSubmConv3d`` (which expects ``(KD, KH, KW, C_in, C_out)``) we apply
``mx.transpose(w, (4, 1, 2, 3, 0))``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from meadow3d.kernels.sparse_subm_conv3d import SparseSubmConv3d


# ---------------------------------------------------------------------------
# Weight reshape helper
# ---------------------------------------------------------------------------


def _npz_conv_to_kernel(weight: mx.array) -> mx.array:
    """Reshape npz Conv3d weight into SparseSubmConv3d's expected layout.

    npz stores  ``(C_out, KH, KW, C_in, KD)``  (output of convert.py's
    ``transpose(0, 2, 3, 4, 1)`` from PT state-dict ``(C_out, KD, KH, KW, C_in)``).

    SparseSubmConv3d.load_weight accepts either ``(K, K, K, C_in, C_out)`` or the
    flat ``(K**3, C_in, C_out)``. We produce the 5D form.
    """
    # (C_out, KH, KW, C_in, KD) -> (KD, KH, KW, C_in, C_out)
    # axis perm: new[0]=KD=old[4], new[1]=KH=old[1], new[2]=KW=old[2],
    #            new[3]=C_in=old[3], new[4]=C_out=old[0]
    return mx.transpose(weight, (4, 1, 2, 3, 0))


# ---------------------------------------------------------------------------
# SparseConvBlock = MLX equivalent of PT SparseResBlock3d
# ---------------------------------------------------------------------------


class SparseConvBlock(nn.Module):
    """Sparse 3D residual block.

    Mirrors ``SparseResBlock3d`` from
    ``sam3d_objects/.../models/structured_latent_flow.py``.

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        emb_channels: Conditioning embedding size (= ``model_channels`` =
                      1024 for slat_flow).
        downsample:   If ``True``, the caller passes already-downsampled
                      ``feats`` and ``coords`` (we have no learnable
                      downsample params; SparseDownsample is mean-pool only).
        upsample:     Symmetric to ``downsample`` for the output stack.
        use_skip:     If ``True`` (and shapes differ) install a ``skip_connection``
                      Linear; otherwise rely on identity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int = 1024,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        assert not (downsample and upsample)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.downsample = downsample
        self.upsample = upsample

        # norm1: LayerNorm32 with affine. Operates on (N, C) feats; equivalent
        # to mlx.nn.LayerNorm with elementwise_affine=True over last dim.
        # (LayerNorm32 in PT just does float32 cast; same numerics in MLX fp32.)
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6, affine=True)

        # norm2: LayerNorm32 with affine=False -> implemented inline
        # (no learnable params).

        # SubMConv3d for conv1 / conv2.
        self.conv1 = SparseSubmConv3d(in_channels, out_channels, kernel_size=3, bias=True)
        self.conv2 = SparseSubmConv3d(out_channels, out_channels, kernel_size=3, bias=True)

        # emb_layers = nn.Sequential(SiLU(), Linear(emb_channels, 2*out_channels)).
        # SiLU has no params; we keep only the Linear, named to match the
        # PT key ``emb_layers.1``.
        self.emb_proj = nn.Linear(emb_channels, 2 * out_channels, bias=True)

        # skip_connection = SparseLinear(in_channels, out_channels) when
        # in != out, else Identity. SparseLinear is just nn.Linear over feats.
        self.has_skip_linear = in_channels != out_channels
        if self.has_skip_linear:
            self.skip_connection = nn.Linear(in_channels, out_channels, bias=True)
        else:
            self.skip_connection = None

    # -- forward --------------------------------------------------------------

    def __call__(
        self,
        feats: mx.array,        # (N, in_channels)
        coords: mx.array,       # (N, 4) int32
        emb: mx.array,          # (B, emb_channels) -- broadcast to all voxels
    ) -> mx.array:
        """Forward pass on already-correct (post-updown) feats/coords.

        Returns ``out_feats`` of shape ``(N, out_channels)``. Coordinates do
        NOT change inside this block (SubMConv3d preserves the coord set);
        the caller is responsible for threading any down/upsample step into
        the ``feats`` and ``coords`` it passes in.
        """
        # PT chunk: emb_layers(emb) -> (B, 2*Cout); split into (scale, shift).
        emb_out = self.emb_proj(nn.silu(emb))           # (B, 2*Cout)
        scale, shift = mx.split(emb_out, 2, axis=-1)    # each (B, Cout)
        # broadcast over voxels: feats are (N, Cout). Per-batch broadcast --
        # since slat inference threads B=1 and feats are unbatched (N, C),
        # we simply select index 0 in the leading axis. (This matches PT;
        # see SLatFlowModelTdfyWrapper.forward where x.shape[0]==1.)
        if scale.ndim == 2:
            # (B, Cout) -> (Cout,) for the typical B=1 case;
            # if B>1 the caller must batch coords with batch index already.
            if scale.shape[0] == 1:
                scale = scale[0]
                shift = shift[0]
            # else: the (B, Cout) shape stays and broadcasts via coords[:, 0]
            #       indexing -- but B>1 isn't exercised in inference here.
        # 1) norm1 + silu + conv1
        h = self.norm1(feats)
        h = nn.silu(h)
        h = self.conv1(h, coords)               # (N, Cout)

        # 2) norm2 (no affine) * (1+scale) + shift
        # LayerNorm over last axis, no learnable params.
        mu = mx.mean(h, axis=-1, keepdims=True)
        var = mx.mean((h - mu) * (h - mu), axis=-1, keepdims=True)
        h = (h - mu) * mx.rsqrt(var + 1e-6)
        h = h * (1.0 + scale) + shift

        # 3) silu + conv2
        h = nn.silu(h)
        h = self.conv2(h, coords)               # (N, Cout)

        # 4) skip
        if self.skip_connection is not None:
            return h + self.skip_connection(feats)
        return h + feats

    # -- weight loading -------------------------------------------------------

    def load_npz(self, w: Dict[str, mx.array], base: str) -> List[str]:
        """Install weights for this block from a flat key dict.

        ``base`` is the per-block prefix, e.g.
        ``"reverse_fn.backbone.input_blocks.0"``. Returns the list of keys
        consumed (for accounting / orphan detection).
        """
        consumed: List[str] = []

        # norm1 (affine LayerNorm)
        self.norm1.weight = w[f"{base}.norm1.weight"]
        self.norm1.bias = w[f"{base}.norm1.bias"]
        consumed += [f"{base}.norm1.weight", f"{base}.norm1.bias"]

        # conv1 / conv2: reshape (C_out, KH, KW, C_in, KD) -> (KD, KH, KW, C_in, C_out)
        wc1 = _npz_conv_to_kernel(w[f"{base}.conv1.conv.weight"])
        bc1 = w[f"{base}.conv1.conv.bias"]
        self.conv1.load_weight(wc1, bc1)
        consumed += [f"{base}.conv1.conv.weight", f"{base}.conv1.conv.bias"]

        wc2 = _npz_conv_to_kernel(w[f"{base}.conv2.conv.weight"])
        bc2 = w[f"{base}.conv2.conv.bias"]
        self.conv2.load_weight(wc2, bc2)
        consumed += [f"{base}.conv2.conv.weight", f"{base}.conv2.conv.bias"]

        # emb_layers.1.* (SiLU is at index 0; Linear at index 1)
        self.emb_proj.weight = w[f"{base}.emb_layers.1.weight"]
        self.emb_proj.bias = w[f"{base}.emb_layers.1.bias"]
        consumed += [f"{base}.emb_layers.1.weight", f"{base}.emb_layers.1.bias"]

        # skip_connection (only when present)
        if self.has_skip_linear:
            self.skip_connection.weight = w[f"{base}.skip_connection.weight"]
            self.skip_connection.bias = w[f"{base}.skip_connection.bias"]
            consumed += [f"{base}.skip_connection.weight", f"{base}.skip_connection.bias"]

        return consumed


# ---------------------------------------------------------------------------
# Stack helpers
# ---------------------------------------------------------------------------


def _enumerate_block_indices(weights: Dict[str, mx.array], prefix: str) -> List[int]:
    """Return sorted list of block indices present under ``prefix``.

    Looks at keys like ``{prefix}<idx>.<rest>``.
    """
    seen: set[int] = set()
    for k in weights:
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        head = rest.split(".", 1)[0]
        try:
            seen.add(int(head))
        except ValueError:
            continue
    return sorted(seen)


def _infer_block_shapes(
    weights: Dict[str, mx.array], prefix: str, idx: int
) -> Tuple[int, int]:
    """Return ``(in_channels, out_channels)`` for block ``idx`` under ``prefix``.

    Reads ``conv1.conv.weight`` shape ``(C_out, KH, KW, C_in, KD)``.
    """
    w = weights[f"{prefix}{idx}.conv1.conv.weight"]
    return int(w.shape[3]), int(w.shape[0])


# ---------------------------------------------------------------------------
# Sparse "input_blocks" stack
# ---------------------------------------------------------------------------


class SparseInputBlocks(nn.Module):
    """Sequence of ``SparseConvBlock`` modules at the DiT input.

    Mirrors PT ``SLatFlowModel.input_blocks`` (an ``nn.ModuleList`` of
    ``SparseResBlock3d``). The last block of each io-channel stage carries
    ``downsample=True``; in slat_flow there is exactly ONE stage and so
    exactly one downsampling block (the LAST one).

    Forward expects callers to thread *post-updown* feats/coords for each
    block. Because slat_flow runs at a single sparse resolution after the
    initial downsample (the block stack is shallow), and the downsample is
    parameter-free, we provide a thin ``forward_with_downsample`` helper
    that lets the caller plug in a downsample function. By default the
    plain ``__call__`` simply iterates blocks assuming feats/coords are
    pre-routed (as in the surrounding tdfy_dit forward).
    """

    def __init__(self, blocks: List[SparseConvBlock], downsample_at: List[int]):
        super().__init__()
        self.blocks = blocks
        # Indices (within self.blocks) where the input has been downsampled.
        # downsample_at[i] == True means block i operates on downsampled coords.
        self.downsample_at = list(downsample_at)

    def __call__(
        self,
        feats: mx.array,
        coords: mx.array,
        emb: mx.array,
    ) -> Tuple[List[mx.array], mx.array, mx.array]:
        """Run all blocks and collect skip features (matches PT impl).

        Returns ``(skips, h_feats, h_coords)`` where ``skips[i]`` is the
        ``feats`` AFTER block i (PT pushes ``h.feats`` to ``skips`` after each
        block). The final ``(h_feats, h_coords)`` is what feeds the DiT
        transformer stack.
        """
        skips: List[mx.array] = []
        h = feats
        c = coords
        for blk in self.blocks:
            h = blk(h, c, emb)
            skips.append(h)
        return skips, h, c

    # -- weight loading -------------------------------------------------------

    @classmethod
    def from_npz(
        cls,
        weights_dict: Dict[str, mx.array],
        prefix: str = "reverse_fn.input_blocks.",
        emb_channels: int = 1024,
    ) -> "SparseInputBlocks":
        """Load all sparse input blocks under ``prefix``.

        ``prefix`` should end with a ``.`` so block indices append cleanly,
        e.g. ``"reverse_fn.backbone.input_blocks."``.
        """
        if not prefix.endswith("."):
            prefix = prefix + "."
        idxs = _enumerate_block_indices(weights_dict, prefix)
        if not idxs:
            raise KeyError(f"No keys found under prefix {prefix!r}")

        blocks: List[SparseConvBlock] = []
        downsample_at: List[bool] = []
        for i, idx in enumerate(idxs):
            in_ch, out_ch = _infer_block_shapes(weights_dict, prefix, idx)
            # In PT, only the LAST block of each io stage is downsample=True.
            # With one io stage (slat_flow) that means the very last block.
            is_last = i == len(idxs) - 1
            blk = SparseConvBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                emb_channels=emb_channels,
                downsample=is_last,
            )
            blocks.append(blk)
            downsample_at.append(is_last)

        m = cls(blocks=blocks, downsample_at=downsample_at)

        consumed: List[str] = []
        for idx, blk in zip(idxs, m.blocks):
            consumed += blk.load_npz(weights_dict, f"{prefix}{idx}")

        # Verify all keys under prefix were consumed.
        all_keys_under_prefix = [k for k in weights_dict if k.startswith(prefix)]
        missing = sorted(set(all_keys_under_prefix) - set(consumed))
        if missing:
            raise KeyError(
                f"Unconsumed keys under {prefix!r}: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} ({len(missing)} total)"
            )

        m._consumed_keys = consumed
        m._prefix = prefix
        return m


# ---------------------------------------------------------------------------
# Sparse "out_blocks" stack
# ---------------------------------------------------------------------------


class SparseOutputBlocks(nn.Module):
    """Sequence of ``SparseConvBlock`` modules at the DiT output.

    Mirrors PT ``SLatFlowModel.out_blocks``. The FIRST block of each io stage
    carries ``upsample=True``; in slat_flow there is one io stage, so the
    first block is the upsampler. ``use_skip_connection=True`` doubles the
    incoming feature width by concatenating with the corresponding
    ``input_blocks`` skip (handled by the caller via
    ``mx.concatenate([h, skip], axis=-1)`` before each block).
    """

    def __init__(self, blocks: List[SparseConvBlock], upsample_at: List[int]):
        super().__init__()
        self.blocks = blocks
        self.upsample_at = list(upsample_at)

    def __call__(
        self,
        feats: mx.array,
        coords: mx.array,
        emb: mx.array,
        skips: List[mx.array],
        use_skip_connection: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        """Run all blocks consuming reversed ``skips``.

        Returns ``(h_feats, h_coords)``. The caller is responsible for
        any updown coord/feat routing (skip features are interpolated to
        the post-upsample resolution outside this module in PT; here we
        assume the caller has aligned them correctly).
        """
        h = feats
        c = coords
        for blk, skip in zip(self.blocks, reversed(skips)):
            if use_skip_connection:
                h = mx.concatenate([h, skip], axis=-1)
            h = blk(h, c, emb)
        return h, c

    # -- weight loading -------------------------------------------------------

    @classmethod
    def from_npz(
        cls,
        weights_dict: Dict[str, mx.array],
        prefix: str = "reverse_fn.out_blocks.",
        emb_channels: int = 1024,
    ) -> "SparseOutputBlocks":
        """Load all sparse output blocks under ``prefix``."""
        if not prefix.endswith("."):
            prefix = prefix + "."
        idxs = _enumerate_block_indices(weights_dict, prefix)
        if not idxs:
            raise KeyError(f"No keys found under prefix {prefix!r}")

        blocks: List[SparseConvBlock] = []
        upsample_at: List[bool] = []
        for i, idx in enumerate(idxs):
            in_ch, out_ch = _infer_block_shapes(weights_dict, prefix, idx)
            # First block of each io stage is upsample=True; with one io
            # stage that's just the first block in the list.
            is_first = i == 0
            blk = SparseConvBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                emb_channels=emb_channels,
                upsample=is_first,
            )
            blocks.append(blk)
            upsample_at.append(is_first)

        m = cls(blocks=blocks, upsample_at=upsample_at)

        consumed: List[str] = []
        for idx, blk in zip(idxs, m.blocks):
            consumed += blk.load_npz(weights_dict, f"{prefix}{idx}")

        all_keys_under_prefix = [k for k in weights_dict if k.startswith(prefix)]
        missing = sorted(set(all_keys_under_prefix) - set(consumed))
        if missing:
            raise KeyError(
                f"Unconsumed keys under {prefix!r}: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} ({len(missing)} total)"
            )

        m._consumed_keys = consumed
        m._prefix = prefix
        return m


__all__ = [
    "SparseConvBlock",
    "SparseInputBlocks",
    "SparseOutputBlocks",
]
