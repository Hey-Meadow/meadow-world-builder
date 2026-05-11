"""Smoke tests for SAM 3D Objects MLX latent_share_transformer port.

Verifies:
1. ``ss_flow.npz`` has **zero** keys under the
   ``latent_share_transformer.*`` prefix (i.e. no learned weights).
2. ``LatentShareTransformer`` merges 5 modalities -> 2 streams with the
   correct token-axis concatenation order.
3. ``LatentSplitTransformer`` slices the merged stream back into 5
   modalities with correct shapes.
4. Round-trip ``split(merge(x)) == x`` for synthetic input.
5. Construction from real npz weights via ``from_npz`` (which is a no-op
   load — the module only needs the merge config).
6. Integration sketch with ``LatentMapping`` to confirm the wiring used by
   the ss_flow inference pipeline (project_dict -> merge -> [DiT stub] ->
   split -> project_dict_out).

Run::
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        meadow_wb/tests/test_latent_share.py
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

from meadow_wb.models.latent_mapping_mlx import LatentMapping  # noqa: E402
from meadow_wb.models.latent_share_mlx import (  # noqa: E402
    SS_FLOW_MERGE_MAP,
    SS_FLOW_TOKEN_LENS,
    LatentShareTransformer,
    LatentSplitTransformer,
)


WEIGHTS_DIR = os.path.join(ROOT, "meadow_wb", "weights", "sam3d_objects")
SS_NPZ = os.path.join(WEIGHTS_DIR, "ss_flow.npz")


# Per-modality (in_dim, token_len) for ss_flow MOT — must match the npz.
SS_MOD = {
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


def _make_synthetic_per_modality(B: int = 2, C: int = 1024, seed: int = 0):
    """Build a dict of (B, token_len, C) latents matching ss_flow MOT shapes."""
    mx.random.seed(seed)
    return {
        n: mx.random.normal((B, SS_MOD[n]["token_len"], C))
        for n in SS_MOD
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_learned_weights_in_npz() -> None:
    _section("ss_flow.npz: latent_share_transformer has 0 learned keys")
    sd = mx.load(SS_NPZ)
    ls_keys = [k for k in sd.keys() if "latent_share_transformer" in k]
    assert len(ls_keys) == 0, (
        f"Expected zero latent_share_transformer.* keys in ss_flow.npz, got "
        f"{len(ls_keys)}: {ls_keys[:5]}"
    )
    print("  OK: 0 latent_share_transformer keys (config-only module).")


def test_default_merge_map_matches_dit_streams() -> None:
    _section("Default merge map -> DiT streams ('shape', '6drotation_normalized')")
    # The DiT was trained with latent_names = [non-merged modalities] + [merged_keys].
    # Streams entering the DiT = SS_MOD - (modalities absorbed in any merged group)
    #                          + list(merge_map.keys()).
    absorbed = {n for names in SS_FLOW_MERGE_MAP.values() for n in names}
    stayed = [n for n in SS_MOD if n not in absorbed]
    streams = stayed + list(SS_FLOW_MERGE_MAP.keys())
    assert sorted(streams) == sorted(["shape", "6drotation_normalized"]), (
        f"DiT streams should be ['shape', '6drotation_normalized'], got "
        f"{sorted(streams)}"
    )
    print(f"  OK: streams={streams}")


def test_forward_merge_shapes() -> None:
    _section("LatentShareTransformer.forward: 5 modalities -> 2 streams")
    merger = LatentShareTransformer()  # default config
    latents = _make_synthetic_per_modality(B=2)
    out = merger(latents)

    assert sorted(out.keys()) == sorted(["shape", "6drotation_normalized"]), (
        f"Expected 2 output streams, got {sorted(out.keys())}"
    )
    # shape: passthrough (B=2, 4096, 1024)
    assert out["shape"].shape == (2, 4096, 1024), out["shape"].shape
    # 6drotation_normalized: concat of 4 single-token modalities -> 4 tokens
    assert out["6drotation_normalized"].shape == (2, 4, 1024), out[
        "6drotation_normalized"
    ].shape
    print(f"  OK: shape={out['shape'].shape}, "
          f"6drotation_normalized={out['6drotation_normalized'].shape}")


def test_concat_order_preserved() -> None:
    _section("LatentShareTransformer: concat order matches merge_map list")
    # Build per-modality tensors with distinct constants so we can read the
    # concat order off the merged tensor's first feature column.
    B, C = 1, 1024
    modal_consts = {
        "6drotation_normalized": 11.0,
        "scale": 22.0,
        "translation": 33.0,
        "translation_scale": 44.0,
    }
    latents = {n: mx.random.normal((B, 4096 if n == "shape" else 1, C))
               for n in SS_MOD}
    for n, c in modal_consts.items():
        latents[n] = mx.full((B, 1, C), c)

    merger = LatentShareTransformer()
    merged = merger(latents)
    rot = merged["6drotation_normalized"]   # (1, 4, 1024)
    # Each token's mean over channel should match the constant we set,
    # in the order of SS_FLOW_MERGE_MAP["6drotation_normalized"].
    means = mx.mean(rot, axis=-1).tolist()[0]
    expected = [
        modal_consts[n] for n in SS_FLOW_MERGE_MAP["6drotation_normalized"]
    ]
    for got, exp, n in zip(
        means, expected, SS_FLOW_MERGE_MAP["6drotation_normalized"]
    ):
        assert abs(got - exp) < 1e-4, (
            f"Token order mismatch at '{n}': got mean={got}, expected {exp}"
        )
    print(f"  OK: order preserved -> "
          f"{SS_FLOW_MERGE_MAP['6drotation_normalized']} -> means={means}")


def test_split_inverse_shapes() -> None:
    _section("LatentSplitTransformer.forward: 2 streams -> 5 modalities")
    splitter = LatentSplitTransformer()
    # Build the DiT-shaped output: same shapes that the DiT would emit.
    merged = {
        "shape": mx.random.normal((2, 4096, 1024)),
        "6drotation_normalized": mx.random.normal((2, 4, 1024)),
    }
    out = splitter(merged)
    assert sorted(out.keys()) == sorted(SS_MOD.keys()), out.keys()
    for n, spec in SS_MOD.items():
        assert out[n].shape == (2, spec["token_len"], 1024), (
            f"{n}: got {out[n].shape}, expected (2, {spec['token_len']}, 1024)"
        )
    print(f"  OK: split into {sorted(out.keys())}")


def test_round_trip_merge_split() -> None:
    _section("Round-trip: split(merge(x)) == x  (no learned ops, exact)")
    merger = LatentShareTransformer()
    splitter = LatentSplitTransformer()
    src = _make_synthetic_per_modality(B=3, seed=42)
    merged = merger(src)
    rec = splitter(merged)
    assert sorted(rec.keys()) == sorted(src.keys()), (sorted(rec.keys()),
                                                       sorted(src.keys()))
    for n in src:
        diff = mx.max(mx.abs(rec[n] - src[n])).item()
        assert diff == 0.0, f"{n}: round-trip diff = {diff} (expected exact 0)"
    print("  OK: exact round-trip on all 5 modalities (B=3).")


def test_from_npz_no_op_load() -> None:
    _section("from_npz: API-symmetric no-op (no learned weights to load)")
    sd = mx.load(SS_NPZ)
    merger = LatentShareTransformer.from_npz(sd)
    splitter = LatentSplitTransformer.from_npz(sd)
    assert merger.merge_map == SS_FLOW_MERGE_MAP
    assert splitter.merge_map == SS_FLOW_MERGE_MAP
    assert splitter.token_lens == SS_FLOW_TOKEN_LENS
    # And they still functionally compose:
    src = _make_synthetic_per_modality(B=1)
    merged = merger(src)
    rec = splitter(merged)
    for n in src:
        assert rec[n].shape == src[n].shape
    print("  OK: from_npz returns canonical config; round-trip shapes match.")


def test_integration_with_latent_mapping() -> None:
    _section("LatentMapping -> LatentShareTransformer wiring sketch")
    lm = LatentMapping.from_npz(SS_NPZ)
    # Build per-modality raw latents at their *true* in_dim (not 1024).
    B = 2
    raw = {
        n: mx.random.normal((B, SS_MOD[n]["token_len"], SS_MOD[n]["in_dim"]))
        for n in SS_MOD
    }
    projected = lm.project_dict(raw)
    # All projected to (B, token_len, 1024)
    for n, spec in SS_MOD.items():
        assert projected[n].shape == (B, spec["token_len"], 1024), (
            n, projected[n].shape
        )
    # Merge using the token_lens read from the constructed LatentMapping.
    token_lens = {n: lm.modalities[n].token_len for n in lm.modality_names}
    merger = LatentShareTransformer()
    splitter = LatentSplitTransformer(token_lens=token_lens)
    merged = merger(projected)
    assert sorted(merged.keys()) == sorted(["shape", "6drotation_normalized"])
    assert merged["shape"].shape == (B, 4096, 1024)
    assert merged["6drotation_normalized"].shape == (B, 4, 1024)
    rec = splitter(merged)
    for n, spec in SS_MOD.items():
        assert rec[n].shape == (B, spec["token_len"], 1024)
    print("  OK: project_dict -> merge -> [DiT] -> split shapes line up.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _main() -> int:
    tests = [
        test_no_learned_weights_in_npz,
        test_default_merge_map_matches_dit_streams,
        test_forward_merge_shapes,
        test_concat_order_preserved,
        test_split_inverse_shapes,
        test_round_trip_merge_split,
        test_from_npz_no_op_load,
        test_integration_with_latent_mapping,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failed == 0:
        print(f"All {len(tests)} tests passed.")
        return 0
    print(f"{failed} of {len(tests)} tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(_main())
