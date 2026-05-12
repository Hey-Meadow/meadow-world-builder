"""Top-level ProPainter inpainter — MLX port.

Mirrors ``model.propainter.InpaintGenerator``. Composes Encoder + SoftSplit
+ feat_prop_module (learnable C=128) + img_prop_module (learnable=False C=3)
+ TemporalSparseTransformerBlock + SoftComp + Decoder.

Provided helpers:
  * ``from_npz(...)``  — instantiate + load main npz.
  * ``forward(masked_frames, completed_flows, masks_in, masks_updated,
                num_local_frames, interpolation='bilinear', t_dilation=2)``
  * ``img_propagation(masked_frames, completed_flows, masks, interpolation)``

External tensor conventions follow PT for the *inputs*:
  * masked_frames:    (B, T, 3, H, W)
  * completed_flows:  tuple of (fwd, bwd) each (B, T-1, 2, H, W)
  * masks_in:         (B, T, 1, H, W)
  * masks_updated:    (B, T, 1, H, W)

The output is also returned in PT layout (B, T_out, 3, H, W) for drop-in
compatibility with downstream code that mimics ``inference_propainter.py``.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .feat_prop import BidirectionalPropagation, flow_warp
from .sparse_transformer import SoftSplit, SoftComp, TemporalSparseTransformerBlock


def _max_pool2d_nhwc(x: mx.array, kernel: tuple[int, int],
                     stride: tuple[int, int], padding: tuple[int, int]) -> mx.array:
    """Channels-last 2D max pool. Matches PT ``nn.MaxPool2d`` semantics."""
    B, H, W, C = x.shape
    kH, kW = kernel
    sH, sW = stride
    pH, pW = padding
    if pH or pW:
        # PT max-pool pads with -inf-ish (-largest finite value)
        x = mx.pad(x, [(0, 0), (pH, pH), (pW, pW), (0, 0)],
                    constant_values=-1e30)
    OH = (x.shape[1] - kH) // sH + 1
    OW = (x.shape[2] - kW) // sW + 1
    # gather windows
    patches = []
    for kh in range(kH):
        for kw in range(kW):
            patches.append(x[:, kh:kh + sH * OH:sH, kw:kw + sW * OW:sW, :])
    stacked = mx.stack(patches, axis=-1)  # (B, OH, OW, C, kH*kW)
    return mx.max(stacked, axis=-1)


def _interp_bilinear_nhwc(x: mx.array, scale_factor: float,
                          align_corners: bool = False) -> mx.array:
    """Bilinear resize an NHWC tensor by ``scale_factor`` (can be < 1).

    Used for ``F.interpolate(..., scale_factor=1/4, mode='bilinear',
    align_corners=False)`` (downsample flows by 4) — match PT semantics.
    """
    from .raft import bilinear_sample_nhwc
    B, H, W, C = x.shape
    new_h = int(round(H * scale_factor))
    new_w = int(round(W * scale_factor))
    if align_corners:
        ys = mx.linspace(0, H - 1, new_h)
        xs = mx.linspace(0, W - 1, new_w)
    else:
        # PT non-align-corners formula:
        #   src = (dst + 0.5) * scale - 0.5
        scale_h = H / new_h
        scale_w = W / new_w
        ys = (mx.arange(new_h).astype(mx.float32) + 0.5) * scale_h - 0.5
        xs = (mx.arange(new_w).astype(mx.float32) + 0.5) * scale_w - 0.5
        # clamp
        ys = mx.clip(ys, 0, H - 1)
        xs = mx.clip(xs, 0, W - 1)
    gy, gx = mx.meshgrid(ys, xs, indexing="ij")
    coords = mx.stack([gx, gy], axis=-1)
    coords = mx.broadcast_to(coords[None], (B, new_h, new_w, 2))
    return bilinear_sample_nhwc(x, coords)


def _interp_nearest_nhwc(x: mx.array, scale_factor: float) -> mx.array:
    B, H, W, C = x.shape
    new_h = int(round(H * scale_factor))
    new_w = int(round(W * scale_factor))
    # PT nearest with non-integer scale uses floor((dst + 0.5) * scale - 0.5) on some
    # versions; for scale_factor=1/4 (downsample) the result is equivalent to
    # taking every 4th pixel.
    if abs(1.0 / scale_factor - round(1.0 / scale_factor)) < 1e-6 and scale_factor < 1.0:
        s = int(round(1.0 / scale_factor))
        return x[:, ::s, ::s, :]
    # general path
    scale_h = H / new_h
    scale_w = W / new_w
    iy = mx.clip(mx.floor((mx.arange(new_h).astype(mx.float32) + 0.5) * scale_h),
                  0, H - 1).astype(mx.int32)
    ix = mx.clip(mx.floor((mx.arange(new_w).astype(mx.float32) + 0.5) * scale_w),
                  0, W - 1).astype(mx.int32)
    out = x[:, iy[:, None], ix[None, :], :]
    return out


class InpaintGenerator(nn.Module):
    """Top-level inpainter."""

    def __init__(self):
        super().__init__()
        channel = 128
        hidden = 512
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        self.t2t_params = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.channel = channel
        self.hidden = hidden

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden, n_head=num_heads, window_size=window_size,
            pool_size=pool_size, depths=depths, t2t_params=self.t2t_params,
        )

    # ---- weight loading ----
    @staticmethod
    def from_npz(npz_path: str | Path) -> "InpaintGenerator":
        m = InpaintGenerator()
        m.load_npz(npz_path)
        return m

    def load_npz(self, npz_path: str | Path):
        data = np.load(str(npz_path))
        flat = {k: mx.array(data[k]) for k in data.files}
        self.load_from_flat(flat)

    def load_from_flat(self, flat: dict[str, mx.array]):
        # encoder
        self.encoder.load_from_flat(flat)
        # decoder
        self.decoder.load_from_flat(flat)
        # ss / sc
        self.ss.embedding.weight = flat["ss.embedding.weight"]
        self.ss.embedding.bias   = flat["ss.embedding.bias"]
        self.sc.embedding.weight = flat["sc.embedding.weight"]
        self.sc.embedding.bias   = flat["sc.embedding.bias"]
        self.sc.bias_conv.weight = flat["sc.bias_conv.weight"]
        self.sc.bias_conv.bias   = flat["sc.bias_conv.bias"]
        # feat_prop_module (learnable, channel=128) — has its own keys
        self.feat_prop_module.load_from_flat(flat, prefix="feat_prop_module.")
        # transformers
        self.transformers.load_from_flat(flat)
        # cast valid_ind_rolled to int32
        for blk in self.transformers.blocks:
            blk.attention.valid_ind_rolled = blk.attention.valid_ind_rolled.astype(mx.int32)

    # ---- helpers to/from PT layout ----
    @staticmethod
    def _pt_to_nhwc_frames(x: mx.array) -> mx.array:
        """(B, T, C, H, W) -> (B, T, H, W, C)."""
        return x.transpose(0, 1, 3, 4, 2)

    @staticmethod
    def _nhwc_to_pt_frames(x: mx.array) -> mx.array:
        return x.transpose(0, 1, 4, 2, 3)

    @staticmethod
    def _pt_to_nhwc_flow(f: mx.array) -> mx.array:
        return f.transpose(0, 1, 3, 4, 2)

    # ---- img_propagation (used before main forward) ----
    def img_propagation(self, masked_frames_pt: mx.array,
                        completed_flows_pt, masks_pt: mx.array,
                        interpolation: str = "nearest"):
        """masked_frames: (B, T, 3, H, W), completed_flows: (fwd, bwd) each
        (B, T-1, 2, H, W), masks: (B, T, 1, H, W)."""
        x = self._pt_to_nhwc_frames(masked_frames_pt)
        ff = self._pt_to_nhwc_flow(completed_flows_pt[0])
        fb = self._pt_to_nhwc_flow(completed_flows_pt[1])
        m = self._pt_to_nhwc_frames(masks_pt)
        _, _, prop, updated_m = self.img_prop_module(x, ff, fb, m, interpolation)
        prop_pt = self._nhwc_to_pt_frames(prop)
        if updated_m is not None:
            updated_m_pt = self._nhwc_to_pt_frames(updated_m)
        else:
            updated_m_pt = None
        return prop_pt, updated_m_pt

    # ---- main forward ----
    def __call__(self, masked_frames_pt: mx.array, completed_flows_pt,
                 masks_in_pt: mx.array, masks_updated_pt: mx.array,
                 num_local_frames: int, interpolation: str = "bilinear",
                 t_dilation: int = 2) -> mx.array:
        """Mirrors PT ``InpaintGenerator.forward`` in eval mode.

        Returns: (B, l_t, 3, H, W).
        """
        l_t = num_local_frames
        B, T, _, ori_h, ori_w = masked_frames_pt.shape

        # ---- encoder ----
        # x: (B*T, ori_h, ori_w, 5) — cat(frames, masks_in, masks_updated) channels
        x_nhwc = self._pt_to_nhwc_frames(masked_frames_pt)  # (B, T, H, W, 3)
        mi_nhwc = self._pt_to_nhwc_frames(masks_in_pt)        # (B, T, H, W, 1)
        mu_nhwc = self._pt_to_nhwc_frames(masks_updated_pt)   # (B, T, H, W, 1)
        enc_in = mx.concatenate([x_nhwc, mi_nhwc, mu_nhwc], axis=-1)  # (B, T, H, W, 5)
        enc_in = enc_in.reshape(B * T, ori_h, ori_w, 5)
        enc_feat = self.encoder(enc_in)  # (B*T, h, w, 128)
        bt, h, w, c = enc_feat.shape
        enc_feat = enc_feat.reshape(B, T, h, w, c)
        local_feat = enc_feat[:, :l_t]
        ref_feat = enc_feat[:, l_t:]
        fold_feat_size = (h, w)

        # ---- downsample flows by 4 ----
        ff = completed_flows_pt[0]  # (B, l_t-1, 2, ori_h, ori_w)
        fb = completed_flows_pt[1]
        ff_nhwc = self._pt_to_nhwc_flow(ff)  # (B, l_t-1, ori_h, ori_w, 2)
        fb_nhwc = self._pt_to_nhwc_flow(fb)
        # flatten time for resize
        Bl, Tf, _, _, _ = ff_nhwc.shape
        ff_flat = ff_nhwc.reshape(Bl * Tf, ori_h, ori_w, 2)
        fb_flat = fb_nhwc.reshape(Bl * Tf, ori_h, ori_w, 2)
        ds_ff = _interp_bilinear_nhwc(ff_flat, 1 / 4, align_corners=False).reshape(Bl, Tf, h, w, 2) / 4.0
        ds_fb = _interp_bilinear_nhwc(fb_flat, 1 / 4, align_corners=False).reshape(Bl, Tf, h, w, 2) / 4.0

        # ds_mask_in (B, T, h, w, 1) nearest 1/4
        mi_flat = mi_nhwc.reshape(B * T, ori_h, ori_w, 1)
        ds_mask_in = _interp_nearest_nhwc(mi_flat, 1 / 4).reshape(B, T, h, w, 1)
        ds_mask_in_local = ds_mask_in[:, :l_t]

        mu_flat_local = mu_nhwc[:, :l_t].reshape(B * l_t, ori_h, ori_w, 1)
        ds_mask_updated_local = _interp_nearest_nhwc(mu_flat_local, 1 / 4).reshape(B, l_t, h, w, 1)

        # mask_pool_l = max_pool(ds_mask_in_local) with kernel=(7,7), stride=(3,3), pad=(3,3)
        mask_pool_in = ds_mask_in_local.reshape(B * l_t, h, w, 1)
        mask_pool = _max_pool2d_nhwc(mask_pool_in, (7, 7), (3, 3), (3, 3))
        mph, mpw = mask_pool.shape[1], mask_pool.shape[2]
        mask_pool_l = mask_pool.reshape(B, l_t, mph, mpw, 1)

        prop_mask_in = mx.concatenate([ds_mask_in_local, ds_mask_updated_local], axis=-1)
        _, _, propagated_local, _ = self.feat_prop_module(
            local_feat, ds_ff, ds_fb, prop_mask_in, interpolation)
        enc_feat = mx.concatenate([propagated_local, ref_feat], axis=1)

        # trans_feat: SoftSplit -> (B, T, f_h, f_w, hidden)
        enc_feat_flat = enc_feat.reshape(B * T, h, w, c)
        trans_feat = self.ss(enc_feat_flat, B, fold_feat_size)
        # transformers: mask_pool_l has shape (B, l_t, mph, mpw, 1) — already NHWC last C
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        # SoftComp: input (B, T, f_h, f_w, hidden) -> (B*T, h, w, c)
        sc_out = self.sc(trans_feat, T, fold_feat_size)
        sc_out = sc_out.reshape(B, T, h, w, c)
        enc_feat = enc_feat + sc_out

        # decoder on first l_t frames only (eval mode)
        dec_in = enc_feat[:, :l_t].reshape(B * l_t, h, w, c)
        out = self.decoder(dec_in, apply_tanh=True)
        out = out.reshape(B, l_t, ori_h, ori_w, 3)
        return self._nhwc_to_pt_frames(out)
