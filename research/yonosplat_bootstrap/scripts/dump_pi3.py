"""Real forward + per-block activation dump of the Pi3 backbone.

Instantiates BackboneLocalGlobal (the actual backbone YoNoSplat re10k uses)
without going through Hydra, loads the matching slice of re10k.ckpt state
dict, registers forward hooks on every block, runs a forward pass on the
2-view test input, and writes one .npz per (block, input/output) pair.

Output: dumps/per_block/<name>.npz with keys 'in_*' and 'out'.

These dumps are the per-block reference signal the parallel MLX-port
agents will assert against.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

UPSTREAM = "/tmp/yonosplat_inspect"
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

REPO_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = REPO_ROOT / "research" / "yonosplat_bootstrap"
WEIGHTS = BOOTSTRAP / "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"
DUMP_DIR = BOOTSTRAP / "dumps"
PER_BLOCK = DUMP_DIR / "per_block"
PER_BLOCK.mkdir(parents=True, exist_ok=True)


def to_np(x):
    """Convert a torch tensor (or tuple thereof) to numpy detached arrays."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (tuple, list)):
        return [to_np(y) for y in x]
    return x


def main():
    # Stub for CUDA rasterizer (used by the wider model; backbone itself doesn't touch it)
    import diff_gaussian_rasterization  # noqa: F401

    # ---- Build backbone --------------------------------------------------------
    from src.model.encoder.backbone.backbone_local_global import (
        BackboneLocalGlobal, BackboneLocalGlobalCfg,
    )

    cfg = BackboneLocalGlobalCfg(
        name="local_global",
        intrinsics_embed_degree=4,
        intrinsics_embed_type="pixelwise",
        predict_intrinsics=True,
        use_pred_intrinsics_for_embed=False,
    )
    print("[dump-pi3] building BackboneLocalGlobal …")
    backbone = BackboneLocalGlobal(
        cfg, d_in=3, pos_type="rope100",
        decoder_size="large", use_checkpoint=False,
    )
    backbone.eval()

    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[dump-pi3] backbone instantiated, {n_params/1e6:.1f} M params")

    # ---- Load matching slice of the YoNoSplat checkpoint -----------------------
    print(f"[dump-pi3] loading {WEIGHTS} …")
    ckpt = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
    full_sd = ckpt["state_dict"]
    # Keys for the backbone live under `encoder.backbone.*`
    bb_sd = {
        k[len("encoder.backbone."):]: v
        for k, v in full_sd.items()
        if k.startswith("encoder.backbone.")
    }
    print(f"[dump-pi3] {len(bb_sd)} backbone tensors found in checkpoint")

    missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)
    print(f"[dump-pi3] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print("  first missing:", missing[:5])
    if unexpected:
        print("  first unexpected:", unexpected[:5])

    # ---- Load test input -------------------------------------------------------
    test = np.load(DUMP_DIR / "test_input.npz")
    images = torch.from_numpy(test["images"])  # (1, 2, 3, 224, 224)
    print(f"[dump-pi3] images = {tuple(images.shape)}")

    # ---- Register per-block hooks ---------------------------------------------
    captured = {}

    def make_hook(name):
        def hook(_mod, inp, out):
            captured[name] = {"in": to_np(inp[0] if isinstance(inp, tuple) and inp else inp),
                              "out": to_np(out)}
        return hook

    hooks = []
    # DINOv2 encoder blocks (24)
    for i, blk in enumerate(backbone.encoder.blocks):
        hooks.append(blk.register_forward_hook(make_hook(f"enc_block_{i:02d}")))
    # Decoder blocks (12, cross-view)
    for i, blk in enumerate(backbone.decoder):
        hooks.append(blk.register_forward_hook(make_hook(f"dec_block_{i:02d}")))
    # Top-level
    hooks.append(backbone.register_forward_hook(make_hook("backbone_full")))
    hooks.append(backbone.encoder.register_forward_hook(make_hook("dino_full")))

    print(f"[dump-pi3] registered {len(hooks)} forward hooks")

    # ---- Forward pass ----------------------------------------------------------
    print("[dump-pi3] running forward (CPU, ~2-3 min for 2-view 224x224) …")
    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        # No intrinsics → use default. forward will compute predicted intrinsics.
        # Need to pass intrinsics shape: (B, V, 3, 3). Use identity 3x3 stack.
        B, V = images.shape[0], images.shape[1]
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)
        try:
            out = backbone(images, intrinsics=intrinsics)
            print(f"[dump-pi3] forward OK in {time.perf_counter()-t0:.1f}s")
            if isinstance(out, tuple):
                for i, t in enumerate(out):
                    print(f"  out[{i}]: {tuple(t.shape) if torch.is_tensor(t) else type(t).__name__}")
        except Exception as e:
            print(f"[dump-pi3] forward FAIL: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return 1

    for h in hooks: h.remove()

    # ---- Save activations -----------------------------------------------------
    n_saved = 0
    for name, tensors in captured.items():
        path = PER_BLOCK / f"{name}.npz"
        save_dict = {}
        if isinstance(tensors["in"], np.ndarray):
            save_dict["in"] = tensors["in"]
        if isinstance(tensors["out"], np.ndarray):
            save_dict["out"] = tensors["out"]
        elif isinstance(tensors["out"], (list, tuple)):
            for i, t in enumerate(tensors["out"]):
                if isinstance(t, np.ndarray):
                    save_dict[f"out_{i}"] = t
        if save_dict:
            np.savez(path, **save_dict)
            n_saved += 1
    print(f"[dump-pi3] wrote {n_saved} per-block .npz files to {PER_BLOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
