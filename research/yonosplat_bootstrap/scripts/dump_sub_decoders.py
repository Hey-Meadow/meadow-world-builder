"""Reference activation dump for the 3 YoNoSplat sub-decoders.

`dump_pi3.py` only dumps the backbone (encoder + decoder). The 3 sub-decoders
(`point_decoder`, `gaussian_decoder`, `camera_decoder`) live one level above
and consume the backbone's already-saved output.

Rather than re-run the full forward (which requires a CUDA xFormers / Pi3
shim we don't have on M1), this script:

  1. Loads the backbone's saved output from `dumps/per_block/backbone_full.npz`
     (= `(hidden, pos, patch_start_idx, x_low, intrinsic_pred)`).
  2. Reconstructs the inputs each sub-decoder receives, mirroring the upstream
     forward exactly (see encoder_yonosplat.py:200-206):
        - `point_decoder(hidden_upsampled, xpos=pos_upsampled)`
        - `gaussian_decoder(hidden_gaussian, xpos=pos_upsampled)`
        - `camera_decoder(hidden, xpos=pos)`
     For the 224x224, upscale_token_ratio=1 setting used in the test input,
     `hidden_upsampled == hidden` and `pos_upsampled == pos`.
     `hidden_gaussian == hidden + rgb_embed(image)`; we likewise capture that
     by re-running the (zero-initialised at training start but finetuned)
     `rgb_embed` Conv from the checkpoint.
  3. Builds 3 `TransformerDecoder` modules from the upstream source, loads the
     matching state-dict slices, and runs `.forward()` on CPU.

Output: `dumps/per_block/sub_<name>.npz` with keys `in`, `xpos`, `out`.

These are the per-decoder reference activations against which the MLX port
(`meadow_sb/models/sub_decoders.py`) is asserted to match within `max abs
diff < 1e-3`.
"""
from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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


def main():
    # Stub for CUDA rasterizer (imported by encoder_yonosplat transitively).
    import diff_gaussian_rasterization  # noqa: F401

    from src.model.encoder.layers.pos_embed import RoPE2D
    from src.model.encoder.layers.transformer_head import TransformerDecoder
    from src.model.encoder.backbone.dinov2.layers import PatchEmbed

    # ---- Backbone dump (input to sub-decoders) --------------------------------
    bb = np.load(PER_BLOCK / "backbone_full.npz")
    hidden = torch.from_numpy(bb["out_0"]).float()          # (2, 261, 2048)
    pos = torch.from_numpy(bb["out_1"]).long()              # (2, 261, 2)
    test = np.load(DUMP_DIR / "test_input.npz")
    images = torch.from_numpy(test["images"]).float()       # (1, 2, 3, 224, 224)
    print(f"[dump-subdec] hidden = {tuple(hidden.shape)}, pos = {tuple(pos.shape)}")

    # ---- Load full checkpoint state dict --------------------------------------
    print(f"[dump-subdec] loading {WEIGHTS} …")
    ckpt = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
    full_sd = ckpt["state_dict"]

    # ---- Build a fresh RoPE2D (matches backbone freq=100) --------------------
    rope = RoPE2D(freq=100.0)

    # ---- Build each sub-decoder ------------------------------------------------
    point_dec = TransformerDecoder(
        in_dim=2048, out_dim=1024, dec_embed_dim=1024, depth=5,
        dec_num_heads=16, mlp_ratio=4.0, rope=rope, use_checkpoint=False,
    )
    gaussian_dec = TransformerDecoder(
        in_dim=2048, out_dim=1024, dec_embed_dim=1024, depth=5,
        dec_num_heads=16, mlp_ratio=4.0, rope=rope, use_checkpoint=False,
    )
    camera_dec = TransformerDecoder(
        in_dim=2048, out_dim=512, dec_embed_dim=1024, depth=5,
        dec_num_heads=16, mlp_ratio=4.0, rope=rope, use_checkpoint=False,
    )

    def slice_sd(prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}

    for dec, prefix in [
        (point_dec,    "encoder.point_decoder."),
        (gaussian_dec, "encoder.gaussian_decoder."),
        (camera_dec,   "encoder.camera_decoder."),
    ]:
        sd = slice_sd(prefix)
        missing, unexpected = dec.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[dump-subdec] {prefix} load: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected")
            if missing:    print("  missing[:5]:", missing[:5])
            if unexpected: print("  unexpected[:5]:", unexpected[:5])
        dec.eval()
    print("[dump-subdec] all three sub-decoders loaded")

    # ---- Build rgb_embed (used to make hidden_gaussian) ----------------------
    # rgb_embed patch_size = backbone.patch_size // upscale_token_ratio = 14 // 2 = 7
    # (state-dict shape (2048, 3, 7, 7) confirms upscale_token_ratio=2 for this ckpt).
    upscale_token_ratio = 2
    patch_size_backbone = 14
    head_patch_size = patch_size_backbone // upscale_token_ratio    # = 7
    norm_layer = partial(nn.LayerNorm, eps=1e-6)
    rgb_embed = PatchEmbed(
        patch_size=head_patch_size, in_chans=3, embed_dim=2048, norm_layer=norm_layer
    )
    rgb_sd = slice_sd("encoder.rgb_embed.")
    missing, unexpected = rgb_embed.load_state_dict(rgb_sd, strict=False)
    if missing or unexpected:
        print(f"[dump-subdec] rgb_embed: {len(missing)} missing, {len(unexpected)} unexpected")
    rgb_embed.eval()

    # ---- Reconstruct inputs ---------------------------------------------------
    # encoder_yonosplat.py:182-202 (with upscale_token_ratio > 1):
    #   hidden_aux = hidden[:, :patch_start_idx]                # (B*V, 5, 2048)
    #   hidden_img = hidden[:, patch_start_idx:]                # (B*V, 256, 2048)
    #   hidden_img = rearrange '(h w) c -> c h w'               # (B*V, 2048, 16, 16)
    #   hidden_img = F.interpolate(..., scale_factor=2)         # (B*V, 2048, 32, 32)
    #   hidden_img = rearrange 'c h w -> (h w) c'               # (B*V, 1024, 2048)
    #   hidden_upsampled = cat(aux, hidden_img)                 # (B*V, 5+1024, 2048)
    #   pos_aux = pos[:, :patch_start_idx]                      # (B*V, 5, 2)
    #   pos_img = PositionGetter(B*V, 32, 32) + 1               # (B*V, 1024, 2)
    #   pos_upsampled = cat(pos_aux, pos_img)                   # (B*V, 5+1024, 2)
    #   rgb_feat = rgb_embed(image)                             # (B*V, 1024, 2048)
    #   hidden_gaussian = hidden_upsampled.clone()
    #   hidden_gaussian[:, patch_start_idx:] += rgb_feat
    #   camera_decoder uses original (non-upsampled) hidden + pos.
    patch_start_idx = 5
    from einops import rearrange
    BV, _N, _C = hidden.shape
    patch_h = patch_w = 224 // patch_size_backbone   # 16

    hidden_aux = hidden[:, :patch_start_idx, :]
    hidden_img = hidden[:, patch_start_idx:, :]
    hidden_img = rearrange(hidden_img, "b (h w) c -> b c h w", h=patch_h, w=patch_w)
    hidden_img = torch.nn.functional.interpolate(
        hidden_img.float(), scale_factor=upscale_token_ratio, mode="bilinear", align_corners=False,
    )
    hidden_img = rearrange(hidden_img, "b c h w -> b (h w) c")
    hidden_upsampled = torch.cat([hidden_aux, hidden_img], dim=1)  # (B*V, 5+1024, 2048)

    # Position getter for upsampled grid (32x32), +1 to leave room for special tokens.
    from src.model.encoder.layers.pos_embed import PositionGetter
    pg = PositionGetter()
    H_up = patch_h * upscale_token_ratio
    W_up = patch_w * upscale_token_ratio
    pos_img = pg(BV, H_up, W_up, device="cpu")                     # (B*V, 1024, 2)
    pos_img = pos_img + 1
    pos_aux = pos[:, :patch_start_idx, :]                           # (B*V, 5, 2)
    pos_upsampled = torch.cat([pos_aux, pos_img], dim=1).long()    # (B*V, 5+1024, 2)
    print(f"[dump-subdec] hidden_upsampled = {tuple(hidden_upsampled.shape)}, "
          f"pos_upsampled = {tuple(pos_upsampled.shape)}")

    rgb = rearrange(images, "b v c h w -> (b v) c h w")            # (2, 3, 224, 224)
    with torch.no_grad():
        rgb_feat = rgb_embed(rgb)                                   # (2, 1024, 2048)
    print(f"[dump-subdec] rgb_feat = {tuple(rgb_feat.shape)}")

    hidden_gaussian = hidden_upsampled.clone()
    hidden_gaussian[:, patch_start_idx:, :] = (
        hidden_gaussian[:, patch_start_idx:, :] + rgb_feat
    )

    # ---- Forward & save -------------------------------------------------------
    cases = [
        ("point",    point_dec,    hidden_upsampled, pos_upsampled),
        ("gaussian", gaussian_dec, hidden_gaussian,  pos_upsampled),
        ("camera",   camera_dec,   hidden,           pos),
    ]
    import time
    for name, dec, h_in, p_in in cases:
        t0 = time.perf_counter()
        with torch.no_grad():
            out = dec(h_in, xpos=p_in)
        dt = time.perf_counter() - t0
        print(f"[dump-subdec] {name}: out={tuple(out.shape)} dtype={out.dtype} "
              f"in {dt:.1f}s, min={out.min().item():.3f}, max={out.max().item():.3f}")
        np.savez(
            PER_BLOCK / f"sub_{name}.npz",
            **{
                "in": h_in.cpu().numpy(),
                "xpos": p_in.cpu().numpy(),
                "out": out.cpu().numpy(),
            },
        )

    print(f"[dump-subdec] done — wrote 3 sub_*.npz files to {PER_BLOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
