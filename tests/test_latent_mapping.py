"""Smoke tests for SAM 3D Objects MLX latent_mapping port.

Tests:
1. Per-modality npz weight discovery for ss_flow (MOT, 5 modalities).
2. Per-modality forward shape correctness — ``shape``, ``6drotation_normalized``,
   ``scale``, ``translation``, ``translation_scale``.
3. Single-modality npz load for slat_flow (no per-modality dict, no pos_emb).
4. ``OutputMapping`` round-trip: input -> DiT-shape -> output back to in_dim.
5. ``PositionalEmbedding.from_npz`` direct table load.

Run::
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        meadow_wb/tests/test_latent_mapping.py
"""

from __future__ import annotations

import os
import sys

import mlx.core as mx

# Repo importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from meadow_wb.models.latent_mapping_mlx import (  # noqa: E402
    LatentMapping,
    OutputMapping,
    PositionalEmbedding,
)


WEIGHTS_DIR = os.path.join(ROOT, "meadow_wb", "weights", "sam3d_objects")
SS_NPZ = os.path.join(WEIGHTS_DIR, "ss_flow.npz")
SLAT_NPZ = os.path.join(WEIGHTS_DIR, "slat_flow.npz")


# Expected modalities + dims for ss_flow MOT:
SS_EXPECTED = {
    "shape": dict(in_dim=8, token_len=4096),
    "6drotation_normalized": dict(in_dim=6, token_len=1),
    "scale": dict(in_dim=3, token_len=1),
    "translation": dict(in_dim=3, token_len=1),
    "translation_scale": dict(in_dim=1, token_len=1),
}


def _section(label: str) -> None:
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ss_flow_mot_load() -> None:
    _section("ss_flow MOT: from_npz discovers all 5 modalities")
    lm = LatentMapping.from_npz(SS_NPZ)
    assert sorted(lm.modality_names) == sorted(SS_EXPECTED.keys()), (
        f"Expected modalities {sorted(SS_EXPECTED.keys())}, got "
        f"{sorted(lm.modality_names)}"
    )
    for name, spec in SS_EXPECTED.items():
        m = lm.modalities[name]
        assert m.in_dim == spec["in_dim"], (
            f"{name}: in_dim={m.in_dim}, expected {spec['in_dim']}"
        )
        assert m.token_len == spec["token_len"], (
            f"{name}: token_len={m.token_len}, expected {spec['token_len']}"
        )
        # weight shape sanity
        assert m.input_layer.weight.shape == (1024, spec["in_dim"]), m.input_layer.weight.shape
        assert m.out_layer.weight.shape == (spec["in_dim"], 1024), m.out_layer.weight.shape
        assert m.pos_emb.shape == (spec["token_len"], 1024), m.pos_emb.shape
        print(
            f"  {name:25s}  in_dim={m.in_dim}  token_len={m.token_len}  "
            f"pos_emb={m.pos_emb.shape}"
        )
    print("  PASS")


def test_ss_flow_mot_forward() -> None:
    _section("ss_flow MOT: forward shape match for all 5 modalities")
    lm = LatentMapping.from_npz(SS_NPZ)
    om = OutputMapping.from_npz(SS_NPZ)

    B = 2
    for name, spec in SS_EXPECTED.items():
        N = spec["token_len"]  # token count must match pos_emb token count
        in_dim = spec["in_dim"]
        x = mx.random.normal((B, N, in_dim))
        h = lm(x, modality=name)
        mx.eval(h)
        assert h.shape == (B, N, 1024), (name, h.shape)

        # round-trip OutputMapping (use h itself as a stand-in for DiT output —
        # we only need to verify shape/dtype, not numerical equivalence).
        y = om(h, modality=name)
        mx.eval(y)
        assert y.shape == (B, N, in_dim), (name, y.shape)
        print(
            f"  {name:25s}  ({B},{N},{in_dim}) -> ({B},{N},1024) -> ({B},{N},{in_dim})  OK"
        )
    print("  PASS")


def test_ss_flow_project_dict() -> None:
    _section("ss_flow MOT: project_dict over multiple modalities")
    lm = LatentMapping.from_npz(SS_NPZ)
    om = OutputMapping.from_npz(SS_NPZ)
    B = 1
    latents = {
        name: mx.random.normal((B, spec["token_len"], spec["in_dim"]))
        for name, spec in SS_EXPECTED.items()
    }
    proj = lm.project_dict(latents)
    for name, spec in SS_EXPECTED.items():
        assert proj[name].shape == (B, spec["token_len"], 1024)
    out = om.project_dict(proj)
    for name, spec in SS_EXPECTED.items():
        assert out[name].shape == (B, spec["token_len"], spec["in_dim"])
    mx.eval(*out.values())
    print("  PASS")


def test_slat_flow_single_modality() -> None:
    _section("slat_flow single-modality: from_npz with prefix=backbone.")
    lm = LatentMapping.from_npz(
        SLAT_NPZ,
        prefix="reverse_fn.backbone.",
        model_channels=128,
    )
    assert lm.modality_names == [""], lm.modality_names
    m = lm.modalities[""]
    # slat input_layer is 8 -> 128 (sparse pre-stage)
    assert m.in_dim == 8, m.in_dim
    assert m.input_layer.weight.shape == (128, 8), m.input_layer.weight.shape
    assert m.out_layer.weight.shape == (8, 128), m.out_layer.weight.shape

    # Forward: no modality keyword should be required (single-modality).
    B, N = 1, 16
    x = mx.random.normal((B, N, 8))
    h = lm(x)  # no modality arg
    mx.eval(h)
    assert h.shape == (B, N, 128), h.shape

    om = OutputMapping.from_npz(
        SLAT_NPZ, prefix="reverse_fn.backbone.", model_channels=128
    )
    y = om(h)
    mx.eval(y)
    assert y.shape == (B, N, 8), y.shape
    print("  PASS")


def test_positional_embedding_direct() -> None:
    _section("PositionalEmbedding.from_npz direct key load")
    pe = PositionalEmbedding.from_npz(
        SS_NPZ, key="reverse_fn.backbone.latent_mapping.shape.pos_emb"
    )
    assert pe.token_len == 4096
    assert pe.model_channels == 1024
    # Apply to a (B, 4096, 1024) tensor
    B = 1
    x = mx.zeros((B, 4096, 1024))
    out = pe(x)
    mx.eval(out)
    assert out.shape == (B, 4096, 1024), out.shape
    # Also test gather form (subset positions)
    pos = mx.array([0, 100, 4095])
    x2 = mx.zeros((B, 3, 1024))
    out2 = pe(x2, positions=pos)
    mx.eval(out2)
    assert out2.shape == (B, 3, 1024), out2.shape
    print("  PASS")


def test_modality_routing_matches_dit() -> None:
    """The ss_flow DiTBackbone (MOT) only routes 'shape' and
    '6drotation_normalized' through transformer blocks; the lower-dim
    modalities (scale, translation, translation_scale) are handled via
    ``latent_share_transformer`` merging at the wrapper level.

    LatentMapping owns all 5 modalities (matches PT
    ``SparseStructureFlowTdfyWrapper.latent_mapping`` ModuleDict)."""
    _section("MOT modality routing: LatentMapping owns 5, DiT owns 2")
    from meadow_wb.models.dit_mlx import MOTDiTBackbone  # noqa: WPS433

    lm = LatentMapping.from_npz(SS_NPZ)
    assert set(lm.modality_names) == set(SS_EXPECTED.keys())
    # DiT default expects 2 modalities
    assert set(MOTDiTBackbone.DEFAULT_LATENT_NAMES) == {
        "shape",
        "6drotation_normalized",
    }
    # The modalities NOT in DiT's set must be handled by the wrapper's
    # latent_share_transformer (out of scope for this agent, but verify the
    # set difference matches expectations).
    extra = set(lm.modality_names) - set(MOTDiTBackbone.DEFAULT_LATENT_NAMES)
    assert extra == {"scale", "translation", "translation_scale"}, extra
    print("  LatentMapping modalities :", sorted(lm.modality_names))
    print("  DiT routed modalities    :", sorted(MOTDiTBackbone.DEFAULT_LATENT_NAMES))
    print("  Wrapper-merged (extra)   :", sorted(extra))
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    test_ss_flow_mot_load()
    test_ss_flow_mot_forward()
    test_ss_flow_project_dict()
    test_slat_flow_single_modality()
    test_positional_embedding_direct()
    test_modality_routing_matches_dit()
    _section("ALL PASS")


if __name__ == "__main__":
    main()
