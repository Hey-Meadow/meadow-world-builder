"""End-to-end pipeline parity test: replicates the inference_propainter.py
flow (RAFT + RFC + main InpaintGenerator) on a small synthetic clip in both
PyTorch and MLX, then compares per-frame max-abs-diff.

This is the closest "PSNR vs PT" measurement we can ship without a real
test video, since synthetic random pixels don't have meaningful PSNR but
do have meaningful max-abs-diff numbers.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ProPainter"))

import torch

# MLX
from propainter_mlx.raft import RAFT as MLXRAFT
from propainter_mlx.flow_completion import RecurrentFlowCompleteNet as MLXRFC
from propainter_mlx.propainter import InpaintGenerator as MLXIG

# PT
from model.modules.flow_comp_raft import RAFT_bi as PTRaftBi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet as PTRFC
from model.propainter import InpaintGenerator as PTIG


def main():
    # ---- build synthetic video ----
    rng = np.random.default_rng(0)
    T = 9
    H, W = 240, 432  # 480p-ish, divisible by 8
    # Synthesise frames with a slowly-moving rectangle
    canvas = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        canvas[t] = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        # also paint a moving rectangle (so flow has something coherent)
        x0 = 20 + t * 3
        y0 = 30
        canvas[t, y0:y0 + 80, x0:x0 + 120] = 200
    frames_np = canvas.astype(np.float32) / 255.0
    frames_pt5d = frames_np.transpose(0, 3, 1, 2)[None]  # (1, T, 3, H, W)
    # to [-1, 1]
    frames_pt5d = frames_pt5d * 2 - 1
    # masks: a static rectangle in the centre
    mask = np.zeros((H, W), dtype=np.float32)
    mask[80:180, 120:300] = 1.0
    masks_np = np.broadcast_to(mask, (T, H, W)).copy()
    masks_pt5d = masks_np[None, :, None]  # (1, T, 1, H, W)

    raft_weights = str(ROOT / "weights/propainter-pt/raft-things.pth")
    rfc_weights = str(ROOT / "weights/propainter-pt/recurrent_flow_completion.pth")
    main_weights = str(ROOT / "weights/propainter-pt/ProPainter.pth")
    mlx_raft = str(ROOT / "weights/propainter-mlx/raft.npz")
    mlx_rfc = str(ROOT / "weights/propainter-mlx/rfc.npz")
    mlx_main = str(ROOT / "weights/propainter-mlx/propainter_main.npz")

    # ============ PyTorch reference ============
    print("[PT] building models...")
    pt_raft = PTRaftBi(model_path=raft_weights, device=torch.device("cpu"))
    pt_raft.eval()
    pt_rfc = PTRFC(model_path=rfc_weights)
    pt_rfc.eval()
    pt_ig = PTIG(init_weights=False)
    pt_ig.load_state_dict(torch.load(main_weights, map_location="cpu", weights_only=True), strict=True)
    pt_ig.eval()

    print("[PT] computing flow + completion...")
    frames_t = torch.from_numpy(frames_pt5d).float()
    flow_masks_t = torch.from_numpy(masks_pt5d).float()
    masks_dil_t = torch.from_numpy(masks_pt5d).float()
    with torch.no_grad():
        gt_flow_f, gt_flow_b = pt_raft(frames_t, iters=20)
        pred_flows_bi_pt, _ = pt_rfc.forward_bidirect_flow((gt_flow_f, gt_flow_b), flow_masks_t)
        pred_flows_bi_pt = pt_rfc.combine_flow((gt_flow_f, gt_flow_b), pred_flows_bi_pt, flow_masks_t)
        masked_frames_t = frames_t * (1 - masks_dil_t)
        prop_imgs, updated_local_masks = pt_ig.img_propagation(masked_frames_t, pred_flows_bi_pt, masks_dil_t, "nearest")
        b, t, _, _, _ = masks_dil_t.size()
        updated_frames_pt = (frames_t * (1 - masks_dil_t)
                              + prop_imgs.view(b, t, 3, H, W) * masks_dil_t)
        updated_masks_pt = updated_local_masks.view(b, t, 1, H, W)

        # one window: l_t = T (no reference), no slicing
        l_t = T
        pred_pt = pt_ig(updated_frames_pt, pred_flows_bi_pt, masks_dil_t,
                          updated_masks_pt, l_t)
    pred_pt_np = pred_pt.numpy()

    # ============ MLX ============
    print("[MLX] building models...")
    raft = MLXRAFT()
    raft.load_npz(mlx_raft)
    rfc = MLXRFC()
    rfc.load_npz(mlx_rfc)
    ig = MLXIG.from_npz(mlx_main)

    print("[MLX] computing flow + completion...")
    t0 = time.time()
    # NHWC for RAFT
    frames_nhwc = mx.array(frames_pt5d[0].transpose(0, 2, 3, 1))  # (T, H, W, 3)
    flows_f_list = []
    flows_b_list = []
    for i in range(T - 1):
        im1 = frames_nhwc[i:i + 1]
        im2 = frames_nhwc[i + 1:i + 2]
        _, ff_up = raft(im1, im2, iters=20)
        _, fb_up = raft(im2, im1, iters=20)
        flows_f_list.append(np.transpose(np.array(ff_up), (0, 3, 1, 2)))
        flows_b_list.append(np.transpose(np.array(fb_up), (0, 3, 1, 2)))
    gt_ff_np = np.concatenate(flows_f_list, axis=0)[None]
    gt_fb_np = np.concatenate(flows_b_list, axis=0)[None]
    gt_ff_mx = mx.array(gt_ff_np)
    gt_fb_mx = mx.array(gt_fb_np)

    flow_masks_mx = mx.array(masks_pt5d)
    masks_dil_mx = mx.array(masks_pt5d)
    frames_mx = mx.array(frames_pt5d)

    pred = rfc.forward_bidirect_flow((gt_ff_mx, gt_fb_mx), flow_masks_mx)
    pred = rfc.combine_flow((gt_ff_mx, gt_fb_mx), pred, flow_masks_mx)
    pf_mx, pb_mx = pred

    masked_frames_mx = frames_mx * (1 - masks_dil_mx)
    prop_imgs_mx, updated_local_mx = ig.img_propagation(masked_frames_mx, (pf_mx, pb_mx), masks_dil_mx, "nearest")
    updated_frames_mx = (frames_mx * (1 - masks_dil_mx) + prop_imgs_mx * masks_dil_mx)
    updated_masks_mx = updated_local_mx

    print("[MLX] running InpaintGenerator...")
    pred_mx = ig(updated_frames_mx, (pf_mx, pb_mx), masks_dil_mx, updated_masks_mx, T)
    mx.eval(pred_mx)
    pred_mlx_np = np.array(pred_mx)
    t_elapsed = time.time() - t0

    diff = np.abs(pred_pt_np - pred_mlx_np)
    print(f"\nPT  output range: [{pred_pt_np.min():.3f}, {pred_pt_np.max():.3f}]")
    print(f"MLX output range: [{pred_mlx_np.min():.3f}, {pred_mlx_np.max():.3f}]")
    print(f"max abs diff: {diff.max():.5f}")
    print(f"mean abs diff: {diff.mean():.5f}")
    # PSNR over [-1, 1] images mapped to [0, 1]
    pt_01 = (pred_pt_np + 1) / 2
    mlx_01 = (pred_mlx_np + 1) / 2
    mse = ((pt_01 - mlx_01) ** 2).mean()
    if mse > 0:
        psnr = -10 * np.log10(mse)
    else:
        psnr = float("inf")
    print(f"PSNR(MLX, PT): {psnr:.2f} dB")
    per_frame_ms = t_elapsed / T * 1000
    print(f"Total e2e elapsed (MLX): {t_elapsed:.2f}s = {per_frame_ms:.0f} ms/frame at {H}x{W}, T={T}")

    gate = 35
    if psnr >= gate:
        print(f"\nPASS (PSNR >= {gate} dB)")
        return 0
    print(f"\nFAIL (PSNR < {gate} dB)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
