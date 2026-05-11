"""Quality-gate tests for `meadow_sb.scripts.convert_weights`.

These tests load the real re10k YoNoSplat checkpoint (3.86 GB; not committed)
and verify:

1. All 8 groups produce the expected number of keys when bucketed.
2. The Conv2d-weight transpose is applied to exactly the three known
   patch-projection tensors (and nothing else 4D).
3. A specific transformer attention weight
   (`encoder.backbone.encoder.blocks.0.attn.qkv.weight`) survives a full
   round-trip into the `dinov2_encoder` npz with its original shape.

Skip behaviour: if the checkpoint file is missing, the tests are skipped
(matches CI environments that don't carry the 3.86 GB weight blob).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

CKPT = os.path.join(
    "research",
    "yonosplat_bootstrap",
    "weights",
    "yonosplat",
    "re10k_224x224_ctx2to32.ckpt",
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(CKPT),
    reason=f"missing checkpoint: {CKPT} (3.86 GB, downloaded on demand)",
)


def _load_module():
    # Imported lazily so the module-level torch import doesn't fire on
    # environments without torch.
    from meadow_sb.scripts import convert_weights as cw
    return cw


def test_group_counts_match_expected():
    """Each of the 8 groups has the expected key count and no orphans."""
    cw = _load_module()
    sd = cw.load_state_dict(CKPT)
    buckets, orphans = cw.bucket_keys(sd)

    assert orphans == [], f"unexpected orphan keys: {orphans[:5]}"
    assert len(sd) == 1222, f"expected 1222 tensors, got {len(sd)}"

    for name, expected in cw.EXPECTED_COUNTS.items():
        actual = len(buckets[name])
        assert actual == expected, (
            f"group {name!r}: expected {expected} keys, got {actual}"
        )

    total = sum(len(v) for v in buckets.values())
    assert total == 1222


def test_conv2d_transpose_only_hits_known_patch_projections():
    """The conv2d transpose must NOT touch register_token / image_mean / image_std."""
    cw = _load_module()
    # Known 4D tensors in the state dict (see dumps/state_dict_tensor_map.json):
    #   encoder.backbone.register_token              (1, 1, 5, 1024)   -- NOT conv
    #   encoder.backbone.image_mean                  (1, 3, 1, 1)      -- NOT conv
    #   encoder.backbone.image_std                   (1, 3, 1, 1)      -- NOT conv
    #   encoder.backbone.encoder.patch_embed.proj.weight        (1024, 3, 14, 14) -- CONV
    #   encoder.backbone.intrinsics_embed_layer.proj.weight     (1024, 25, 14, 14) -- CONV
    #   encoder.rgb_embed.proj.weight                           (2048, 3, 7, 7)   -- CONV
    assert cw._is_conv2d_weight("encoder.backbone.encoder.patch_embed.proj.weight")
    assert cw._is_conv2d_weight("encoder.backbone.intrinsics_embed_layer.proj.weight")
    assert cw._is_conv2d_weight("encoder.rgb_embed.proj.weight")
    assert not cw._is_conv2d_weight("encoder.backbone.register_token")
    assert not cw._is_conv2d_weight("encoder.backbone.image_mean")
    assert not cw._is_conv2d_weight("encoder.backbone.image_std")
    # Linear / norm keys must not be flagged either.
    assert not cw._is_conv2d_weight(
        "encoder.backbone.encoder.blocks.0.attn.qkv.weight"
    )


def test_qkv_weight_roundtrip_into_dinov2_encoder_npz(tmp_path):
    """`...blocks.0.attn.qkv.weight` round-trips with original (3072, 1024) shape."""
    cw = _load_module()
    out_dir = str(tmp_path / "weights")
    cw.convert_all(CKPT, out_dir, no_compress=True)

    npz_path = os.path.join(out_dir, "dinov2_encoder.npz")
    assert os.path.exists(npz_path), f"missing: {npz_path}"
    with np.load(npz_path) as data:
        key = "encoder.backbone.encoder.blocks.0.attn.qkv.weight"
        assert key in data.files, (
            f"qkv key missing from dinov2_encoder.npz; "
            f"first 5 keys: {data.files[:5]}"
        )
        arr = data[key]
        # qkv weight is a 2D Linear (out=3*hidden=3072, in=1024); no transpose.
        assert arr.shape == (3072, 1024), f"unexpected shape: {arr.shape}"
        assert arr.dtype == np.float32, f"unexpected dtype: {arr.dtype}"


def test_patch_embed_transpose_applied(tmp_path):
    """`patch_embed.proj.weight` gets the (O,I,H,W)->(O,H,W,I) transpose."""
    cw = _load_module()
    out_dir = str(tmp_path / "weights")
    cw.convert_all(CKPT, out_dir, no_compress=True)

    npz_path = os.path.join(out_dir, "dinov2_encoder.npz")
    with np.load(npz_path) as data:
        key = "encoder.backbone.encoder.patch_embed.proj.weight"
        assert key in data.files
        # Source shape (1024, 3, 14, 14) -> MLX (1024, 14, 14, 3).
        assert data[key].shape == (1024, 14, 14, 3), (
            f"patch_embed.proj.weight not transposed; got {data[key].shape}"
        )


def test_register_token_4d_NOT_transposed(tmp_path):
    """register_token / image_mean / image_std are 4D but must be left alone."""
    cw = _load_module()
    out_dir = str(tmp_path / "weights")
    cw.convert_all(CKPT, out_dir, no_compress=True)

    npz_path = os.path.join(out_dir, "register_token.npz")
    with np.load(npz_path) as data:
        assert data["encoder.backbone.register_token"].shape == (1, 1, 5, 1024)
        assert data["encoder.backbone.image_mean"].shape == (1, 3, 1, 1)
        assert data["encoder.backbone.image_std"].shape == (1, 3, 1, 1)
