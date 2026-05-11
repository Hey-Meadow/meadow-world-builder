# meadow_sb — Meadow Space Builder (YoNoSplat MLX port)

Sibling of `meadow_wb` (Meadow World Builder for single-image objects).
Targets the multi-image, scene-scale 3DGS + camera-pose path.

Status: **alpha bootstrap** — modules being ported by parallel agents.
See `docs/PORT_PLAN_YONOSPLAT.md` and `docs/YONOSPLAT_INTERFACE_CONTRACT.md`.

## Module ownership (parallel sprint)

| File | Owner agent | Status |
|---|---|---|
| `models/dinov2_encoder.py` | A | in progress |
| `models/croco_decoder.py` | B | in progress |
| `models/sub_decoders.py` | C | in progress |
| `models/heads.py` | D | in progress |
| `models/gaussian_adapter.py` | E | in progress |
| `models/rasterizer.py` | F (Tier-1 gsplat wrapper) | in progress |
| `scripts/convert_weights.py` | G | in progress |
| `scripts/e2e_test.py` + `tests/` | H | in progress |
