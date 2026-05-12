"""End-to-end video inpainting CLI for the MLX ProPainter port.

Mirrors ``inference_propainter.py`` from upstream. Inputs:
  --video  : a video file (mp4/mov/avi) OR a folder of frames
  --mask   : a folder of per-frame mask PNGs (or a single PNG)
  --out    : output mp4 path
Optional:
  --neighbor_length, --ref_stride, --subvideo_length, --raft_iter
  --mask_dilation
  --height, --width, --resize_ratio  : pre-resize
  --save_fps                          : output fps
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mlx.core as mx
from PIL import Image
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from propainter_mlx.raft import RAFT
from propainter_mlx.flow_completion import RecurrentFlowCompleteNet
from propainter_mlx.propainter import InpaintGenerator


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _binary_dilation(mask: np.ndarray, iters: int) -> np.ndarray:
    """Square-kernel binary dilation. Implements scipy.ndimage.binary_dilation
    with default 3x3 cross structure iterated ``iters`` times."""
    try:
        import scipy.ndimage
        return scipy.ndimage.binary_dilation(mask, iterations=iters).astype(np.uint8)
    except ImportError:
        m = mask.astype(np.uint8)
        for _ in range(iters):
            # 3x3 max filter
            pad = np.pad(m, 1, mode="edge")
            m = np.maximum.reduce([
                pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:],
                pad[1:-1, 1:-1],
            ])
        return m


def read_frames(path: str) -> Tuple[List[Image.Image], int, str]:
    """Returns (frames, fps, video_name)."""
    if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        fps = int(round(meta.get("fps", 24)))
        frames = [Image.fromarray(f) for f in reader]
        reader.close()
        name = Path(path).stem
    else:
        d = Path(path)
        names = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        frames = []
        for p in names:
            frames.append(Image.open(p).convert("RGB"))
        fps = 24
        name = d.name
    return frames, fps, name


def read_masks(path: str, n_frames: int, target_size: Tuple[int, int],
                dilation: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Returns (flow_masks, masks_dilated) as lists of uint8 HxW arrays in {0, 255}.
    target_size is (W, H) PIL-order."""
    masks_img: List[Image.Image] = []
    if path.lower().endswith((".png", ".jpg", ".jpeg")):
        masks_img = [Image.open(path)]
    else:
        d = Path(path)
        names = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for p in names:
            masks_img.append(Image.open(p))

    flow_masks: List[np.ndarray] = []
    masks_dilated: List[np.ndarray] = []
    for mi in masks_img:
        if target_size is not None:
            mi = mi.resize(target_size, Image.NEAREST)
        m = np.array(mi.convert("L"))
        if dilation > 0:
            flow_m = _binary_dilation(m > 0, dilation) * 255
            dil_m = _binary_dilation(m > 0, dilation) * 255
        else:
            flow_m = ((m > 0).astype(np.uint8)) * 255
            dil_m = flow_m
        flow_masks.append(flow_m.astype(np.uint8))
        masks_dilated.append(dil_m.astype(np.uint8))
    if len(masks_img) == 1:
        flow_masks = flow_masks * n_frames
        masks_dilated = masks_dilated * n_frames
    assert len(flow_masks) >= n_frames, f"mask count {len(flow_masks)} < frames {n_frames}"
    return flow_masks[:n_frames], masks_dilated[:n_frames]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def to_mx_pt5d(np_arr: np.ndarray) -> mx.array:
    """np (T, C, H, W) -> mx (1, T, C, H, W)."""
    return mx.array(np_arr[None])


def get_ref_index(mid: int, neighbor_ids: List[int], length: int,
                  ref_stride: int = 10, ref_num: int = -1) -> List[int]:
    ref = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref.append(i)
    else:
        start = max(0, mid - ref_stride * (ref_num // 2))
        end = min(length, mid + ref_stride * (ref_num // 2))
        for i in range(start, end, ref_stride):
            if i not in neighbor_ids:
                if len(ref) > ref_num:
                    break
                ref.append(i)
    return ref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--video", required=True)
    p.add_argument("-m", "--mask", required=True)
    p.add_argument("-o", "--out", default="inpaint_out.mp4")
    p.add_argument("--resize_ratio", type=float, default=1.0)
    p.add_argument("--height", type=int, default=-1)
    p.add_argument("--width", type=int, default=-1)
    p.add_argument("--mask_dilation", type=int, default=4)
    p.add_argument("--ref_stride", type=int, default=10)
    p.add_argument("--neighbor_length", type=int, default=10)
    p.add_argument("--subvideo_length", type=int, default=80)
    p.add_argument("--raft_iter", type=int, default=20)
    p.add_argument("--save_fps", type=int, default=None)
    p.add_argument("--weights-dir", default=str(ROOT / "weights/propainter-mlx"))
    args = p.parse_args()

    # ---- load video ----
    frames_pil, fps, video_name = read_frames(args.video)
    n_frames = len(frames_pil)
    print(f"Loaded {n_frames} frames from {args.video}")

    if args.width != -1 and args.height != -1:
        target_size = (args.width, args.height)
    elif args.resize_ratio != 1.0:
        ow, oh = frames_pil[0].size
        target_size = (int(args.resize_ratio * ow), int(args.resize_ratio * oh))
    else:
        target_size = frames_pil[0].size
    # snap to multiple of 8
    target_size = (target_size[0] - target_size[0] % 8,
                    target_size[1] - target_size[1] % 8)
    print(f"Target processing size (W, H): {target_size}")
    frames_pil = [f.resize(target_size) for f in frames_pil]

    flow_masks_np, masks_dilated_np = read_masks(args.mask, n_frames, target_size, args.mask_dilation)

    out_w, out_h = target_size

    # ---- convert frames + masks to MLX tensors ----
    # frames: (T, 3, H, W) float32 in [-1, 1]
    frames_np = np.stack([np.array(f).astype(np.float32) for f in frames_pil], axis=0) / 255.0
    frames_np = (frames_np * 2 - 1).transpose(0, 3, 1, 2)
    flow_masks_arr = np.stack([m[None].astype(np.float32) / 255.0 for m in flow_masks_np], axis=0)
    masks_dilated_arr = np.stack([m[None].astype(np.float32) / 255.0 for m in masks_dilated_np], axis=0)

    frames_mx = to_mx_pt5d(frames_np)          # (1, T, 3, H, W)
    flow_masks_mx = to_mx_pt5d(flow_masks_arr)  # (1, T, 1, H, W)
    masks_dil_mx = to_mx_pt5d(masks_dilated_arr)

    # ---- load models ----
    weights = Path(args.weights_dir)
    print("Loading RAFT...")
    raft = RAFT()
    raft.load_npz(str(weights / "raft.npz"))
    print("Loading flow completion...")
    rfc = RecurrentFlowCompleteNet()
    rfc.load_npz(str(weights / "rfc.npz"))
    print("Loading ProPainter main...")
    inpainter = InpaintGenerator.from_npz(str(weights / "propainter_main.npz"))
    print("All models loaded.")

    # ---- 1. compute optical flow (forward + backward) ----
    print("[1/3] computing optical flow...")
    t0 = time.time()
    # convert each frame to NHWC for RAFT
    frames_nhwc = mx.array(np.transpose((frames_np + 1) * 0.5 * 2 - 1, (0, 2, 3, 1)))  # back to [-1,1] NHWC
    # NB: RAFT expects (B, H, W, 3) in [-1, 1] per the conversion notes.
    # Build pairs
    gt_flows_f = []
    gt_flows_b = []
    short = 12 if out_w <= 640 else (8 if out_w <= 720 else 4)
    for s in range(0, n_frames, short):
        e = min(n_frames, s + short)
        clip = frames_nhwc[s:e] if s == 0 else frames_nhwc[s - 1:e]
        # pair (i, i+1)
        for i in range(clip.shape[0] - 1):
            im1 = clip[i:i + 1]
            im2 = clip[i + 1:i + 2]
            _, flow_f = raft(im1, im2, iters=args.raft_iter)
            _, flow_b = raft(im2, im1, iters=args.raft_iter)
            mx.eval(flow_f, flow_b)
            # convert to (1, 2, H, W) PT layout
            gt_flows_f.append(np.transpose(np.array(flow_f), (0, 3, 1, 2)))
            gt_flows_b.append(np.transpose(np.array(flow_b), (0, 3, 1, 2)))
    gt_flows_f_np = np.concatenate(gt_flows_f, axis=0)[None]  # (1, T-1, 2, H, W)
    gt_flows_b_np = np.concatenate(gt_flows_b, axis=0)[None]
    print(f"   raft done in {time.time()-t0:.1f}s, flow shape {gt_flows_f_np.shape}")
    gt_flows_f_mx = mx.array(gt_flows_f_np)
    gt_flows_b_mx = mx.array(gt_flows_b_np)

    # ---- 2. complete flow ----
    print("[2/3] completing flow...")
    t0 = time.time()
    flow_len = gt_flows_f_mx.shape[1]
    if flow_len > args.subvideo_length:
        pf_list, pb_list = [], []
        pad = 5
        for f in range(0, flow_len, args.subvideo_length):
            s_f = max(0, f - pad)
            e_f = min(flow_len, f + args.subvideo_length + pad)
            pad_s = max(0, f) - s_f
            pad_e = e_f - min(flow_len, f + args.subvideo_length)
            sub_in = (gt_flows_f_mx[:, s_f:e_f], gt_flows_b_mx[:, s_f:e_f])
            pred_sub = rfc.forward_bidirect_flow(sub_in, flow_masks_mx[:, s_f:e_f + 1])
            pred_sub = rfc.combine_flow(sub_in, pred_sub, flow_masks_mx[:, s_f:e_f + 1])
            pf_list.append(pred_sub[0][:, pad_s:e_f - s_f - pad_e])
            pb_list.append(pred_sub[1][:, pad_s:e_f - s_f - pad_e])
        pred_fwd = mx.concatenate(pf_list, axis=1)
        pred_bwd = mx.concatenate(pb_list, axis=1)
    else:
        pred = rfc.forward_bidirect_flow((gt_flows_f_mx, gt_flows_b_mx), flow_masks_mx)
        pred = rfc.combine_flow((gt_flows_f_mx, gt_flows_b_mx), pred, flow_masks_mx)
        pred_fwd, pred_bwd = pred
    mx.eval(pred_fwd, pred_bwd)
    print(f"   rfc done in {time.time()-t0:.1f}s")

    # ---- 3. image propagation ----
    print("[3/3] image propagation + main inpaint...")
    t0 = time.time()
    masked_frames = frames_mx * (1 - masks_dil_mx)
    sub_img_len = min(100, args.subvideo_length)
    if n_frames > sub_img_len:
        updated_f_list, updated_m_list = [], []
        pad = 10
        for f in range(0, n_frames, sub_img_len):
            s_f = max(0, f - pad)
            e_f = min(n_frames, f + sub_img_len + pad)
            pad_s = max(0, f) - s_f
            pad_e = e_f - min(n_frames, f + sub_img_len)
            sub_flows = (pred_fwd[:, s_f:e_f - 1], pred_bwd[:, s_f:e_f - 1])
            prop_imgs, updated_local_masks = inpainter.img_propagation(
                masked_frames[:, s_f:e_f], sub_flows, masks_dil_mx[:, s_f:e_f], "nearest")
            updated_f_sub = (frames_mx[:, s_f:e_f] * (1 - masks_dil_mx[:, s_f:e_f])
                              + prop_imgs * masks_dil_mx[:, s_f:e_f])
            updated_f_list.append(updated_f_sub[:, pad_s:e_f - s_f - pad_e])
            updated_m_list.append(updated_local_masks[:, pad_s:e_f - s_f - pad_e])
        updated_frames = mx.concatenate(updated_f_list, axis=1)
        updated_masks = mx.concatenate(updated_m_list, axis=1)
    else:
        prop_imgs, updated_local_masks = inpainter.img_propagation(
            masked_frames, (pred_fwd, pred_bwd), masks_dil_mx, "nearest")
        updated_frames = (frames_mx * (1 - masks_dil_mx)
                           + prop_imgs * masks_dil_mx)
        updated_masks = updated_local_masks
    mx.eval(updated_frames, updated_masks)

    # ---- 4. feature propagation + transformer sliding window ----
    comp_frames: List[np.ndarray | None] = [None] * n_frames
    neighbor_stride = args.neighbor_length // 2
    ref_num = (args.subvideo_length // args.ref_stride) if n_frames > args.subvideo_length else -1
    ori_frames = [np.array(f).astype(np.uint8) for f in frames_pil]
    binary_masks = [(m[..., None].astype(np.uint8) // 255).repeat(3, axis=-1) for m in masks_dilated_np]
    n_steps = 0
    for f in range(0, n_frames, neighbor_stride):
        neighbor_ids = list(range(max(0, f - neighbor_stride),
                                    min(n_frames, f + neighbor_stride + 1)))
        ref_ids = get_ref_index(f, neighbor_ids, n_frames, args.ref_stride, ref_num)
        all_ids = neighbor_ids + ref_ids
        l_t = len(neighbor_ids)

        sel_imgs = updated_frames[:, all_ids]
        sel_masks_in = masks_dil_mx[:, all_ids]
        sel_masks_up = updated_masks[:, all_ids]
        sel_flows = (pred_fwd[:, neighbor_ids[:-1]], pred_bwd[:, neighbor_ids[:-1]])

        pred_img = inpainter(sel_imgs, sel_flows, sel_masks_in, sel_masks_up, l_t)
        mx.eval(pred_img)
        pred_np = np.array(pred_img)  # (1, l_t, 3, H, W)
        pred_np = (pred_np + 1) / 2
        pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
        pred_np = np.transpose(pred_np[0], (0, 2, 3, 1))  # (l_t, H, W, 3)

        for i, idx in enumerate(neighbor_ids):
            bm = binary_masks[idx]
            inpainted = pred_np[i] * bm + ori_frames[idx] * (1 - bm)
            if comp_frames[idx] is None:
                comp_frames[idx] = inpainted.astype(np.uint8)
            else:
                comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5
                                      + inpainted.astype(np.float32) * 0.5).astype(np.uint8)
        n_steps += 1
    print(f"   main inpaint done in {time.time()-t0:.1f}s ({n_steps} sliding windows)")

    # ---- save output video ----
    save_fps = args.save_fps or fps
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), comp_frames, fps=save_fps, quality=7)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
