"""Temporal sparse transformer — MLX port.

Mirrors ``model.modules.sparse_transformer`` from upstream ProPainter.
The whole module reproduces:

  * ``SoftSplit``           — F.unfold(k=7,s=3,p=3) + Linear (channels-last)
  * ``SoftComp``            — Linear + F.fold(...) + 3x3 Conv (channels-last)
  * ``FusionFeedForward``   — fc1 -> overlap-add unfold/fold with divisor
                              compensation -> fc2
  * ``SparseWindowAttention``       — single attention block (window + pool
                                       token + rolled-window mixing + sparse
                                       mask gated by ``mask_pool_l``)
  * ``TemporalSparseTransformer``   — pre-norm attention + FFN
  * ``TemporalSparseTransformerBlock`` — N blocks with t-dilation

NHWC throughout. Token tensors carry the shape ``(B, T, h, w, C)`` where
``h, w`` are the post-SoftSplit token-grid dimensions (``= (H+2p-k)/s + 1``
for spatial input size ``H``).
"""
from __future__ import annotations
import math
from typing import Sequence
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# unfold / fold (NHWC) — manual gather + scatter-add via boolean accumulate
# ---------------------------------------------------------------------------

def unfold_nhwc(x: mx.array, kernel: tuple[int, int], stride: tuple[int, int],
                padding: tuple[int, int]) -> mx.array:
    """``F.unfold`` for NHWC inputs.

    Args:
        x:        (N, H, W, C)
        kernel:   (kH, kW)
        stride:   (sH, sW)
        padding:  (pH, pW)
    Returns:
        patches:  (N, OH*OW, C*kH*kW)  — the "L" axis matches PT's
                  L = OH * OW with row-major (oh outer, ow inner).
                  The last axis is in PT's ``F.unfold`` order:
                  ``(c, kh, kw)`` with ``c`` outermost / ``kw`` innermost.
                  This matches the column-ordering of
                  ``F.unfold(...).permute(0, 2, 1)``.
    """
    N, H, W, C = x.shape
    kH, kW = kernel
    sH, sW = stride
    pH, pW = padding
    OH = (H + 2 * pH - kH) // sH + 1
    OW = (W + 2 * pW - kW) // sW + 1

    if pH or pW:
        x = mx.pad(x, [(0, 0), (pH, pH), (pW, pW), (0, 0)])
    # Collect kH*kW shifted views, each (N, OH, OW, C), then stack
    # along a new "kernel" axis of size kH*kW.
    patches = []
    for kh in range(kH):
        for kw in range(kW):
            patch = x[:, kh:kh + sH * OH:sH, kw:kw + sW * OW:sW, :]
            patches.append(patch)
    # stack -> (N, OH, OW, kH*kW, C); we want (kh, kw, c) inner stride
    out = mx.stack(patches, axis=3)  # (N, OH, OW, kH*kW, C)
    # Reorder last two axes to put C outer, (kh, kw) inner → matches PT.
    out = out.transpose(0, 1, 2, 4, 3)  # (N, OH, OW, C, kH*kW)
    out = out.reshape(N, OH * OW, C * kH * kW)
    return out


def fold_nhwc(patches: mx.array, output_size: tuple[int, int],
              kernel: tuple[int, int], stride: tuple[int, int],
              padding: tuple[int, int], channels: int) -> mx.array:
    """``F.fold`` for NHWC outputs.

    Args:
        patches:     (N, OH*OW, C*kH*kW) — PT inner ordering ``(c, kh, kw)``.
        output_size: (H, W)
        kernel, stride, padding: as in ``F.fold``.
        channels:    C (needed to disambiguate the last-dim split).
    Returns:
        (N, H, W, C) — overlap-summed.
    """
    H, W = output_size
    kH, kW = kernel
    sH, sW = stride
    pH, pW = padding
    OH = (H + 2 * pH - kH) // sH + 1
    OW = (W + 2 * pW - kW) // sW + 1
    N = patches.shape[0]
    C = channels

    # (N, OH, OW, C, kH, kW) — PT inner ordering
    p = patches.reshape(N, OH, OW, C, kH, kW)
    # convert to (N, OH, OW, kH, kW, C) for the gather/scatter loop
    p = p.transpose(0, 1, 2, 4, 5, 3)

    Hp = H + 2 * pH
    Wp = W + 2 * pW
    # Materialise padded canvas and accumulate.
    out = mx.zeros((N, Hp, Wp, C), dtype=patches.dtype)
    for kh in range(kH):
        for kw in range(kW):
            slab = p[:, :, :, kh, kw, :]  # (N, OH, OW, C)
            # Build a sparse pad/scatter via mx.pad of slab into the right
            # offset, then add. Equivalent to:
            #   out[:, kh:kh+sH*OH:sH, kw:kw+sW*OW:sW, :] += slab
            # which MLX doesn't support directly; instead we lay slab out on
            # a (Hp, Wp) canvas at the proper stride and add.
            # We achieve the stride by interleaving zeros with mx.zeros and
            # indexed assignment; cleanest portable approach: build the
            # canvas via `mx.pad` after upsampling slab to (sH*OH, sW*OW).
            # Upsample via reshape+pad with zeros.
            up = mx.zeros((N, sH * OH, sW * OW, C), dtype=patches.dtype)
            # Construct an interleaved canvas:
            up = up.reshape(N, OH, sH, OW, sW, C)
            # Place slab at (0, 0) inside each stride cell
            # Use scatter via concatenation. The simplest: pad each row of
            # slab on the right by (sH-1) zeros, then reshape.
            slab_e = slab[:, :, None, :, None, :]  # (N,OH,1,OW,1,C)
            zeros_h = mx.zeros((N, OH, sH - 1, OW, 1, C), dtype=patches.dtype) if sH > 1 else None
            zeros_w = mx.zeros((N, OH, sH, OW, sW - 1, C), dtype=patches.dtype) if sW > 1 else None
            if zeros_h is not None:
                slab_e = mx.concatenate([slab_e, zeros_h], axis=2)
            if zeros_w is not None:
                slab_e = mx.concatenate([slab_e, zeros_w], axis=4)
            up = slab_e.reshape(N, sH * OH, sW * OW, C)
            # Pad up so it lands at position (kh, kw) and fills Hp x Wp.
            pad_top = kh
            pad_bot = Hp - kh - sH * OH
            pad_left = kw
            pad_right = Wp - kw - sW * OW
            # Pad/crop to (Hp, Wp).
            if pad_bot < 0 or pad_right < 0:
                up = up[:, :Hp - pad_top, :Wp - pad_left, :]
                pad_bot = max(0, Hp - pad_top - up.shape[1])
                pad_right = max(0, Wp - pad_left - up.shape[2])
            up = mx.pad(up, [(0, 0), (pad_top, pad_bot),
                              (pad_left, pad_right), (0, 0)])
            out = out + up

    # Crop padding back off
    out = out[:, pH:pH + H, pW:pW + W, :]
    return out


# ---------------------------------------------------------------------------
# SoftSplit / SoftComp / FusionFeedForward
# ---------------------------------------------------------------------------

def _grid_size(output_size: tuple[int, int], kernel, stride, padding):
    H, W = output_size
    kH, kW = kernel
    sH, sW = stride
    pH, pW = padding
    f_h = (H + 2 * pH - (kH - 1) - 1) // sH + 1
    f_w = (W + 2 * pW - (kW - 1) - 1) // sW + 1
    return f_h, f_w


class SoftSplit(nn.Module):
    """Unfold kxk patches and linearly project to ``hidden``."""
    def __init__(self, channel: int, hidden: int, kernel: tuple[int, int],
                 stride: tuple[int, int], padding: tuple[int, int]):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        c_in = kernel[0] * kernel[1] * channel
        self.embedding = nn.Linear(c_in, hidden)

    def __call__(self, x: mx.array, b: int, output_size: tuple[int, int]) -> mx.array:
        """x: (b*t, H, W, C). Returns (b, t, f_h, f_w, hidden)."""
        f_h, f_w = _grid_size(output_size, self.kernel, self.stride, self.padding)
        feat = unfold_nhwc(x, self.kernel, self.stride, self.padding)  # (b*t, L, k*k*C)
        feat = self.embedding(feat)  # (b*t, L, hidden)
        out = feat.reshape(b, -1, f_h, f_w, feat.shape[-1])
        return out


class SoftComp(nn.Module):
    """Linear project to ``k*k*channel`` then fold back to (H, W, channel)."""
    def __init__(self, channel: int, hidden: int, kernel: tuple[int, int],
                 stride: tuple[int, int], padding: tuple[int, int]):
        super().__init__()
        self.channel = channel
        c_out = kernel[0] * kernel[1] * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel, channel, 3, stride=1, padding=1)

    def __call__(self, x: mx.array, t: int, output_size: tuple[int, int]) -> mx.array:
        """x: (b, T, f_h, f_w, hidden). Returns (b*t, H, W, channel)."""
        b_, _, _, _, c_ = x.shape
        x = x.reshape(b_, -1, c_)
        feat = self.embedding(x)  # (b, T*L, k*k*C)
        b, _, c = feat.shape
        feat = feat.reshape(b * t, -1, c)
        feat = fold_nhwc(feat, output_size, self.kernel, self.stride, self.padding, self.channel)
        # bias_conv operates on (N, H, W, C)
        feat = self.bias_conv(feat)
        return feat


class FusionFeedForward(nn.Module):
    """Token-overlap MLP with fold-divisor compensation."""
    def __init__(self, dim: int, hidden_dim: int = 1960,
                 t2t_params: dict | None = None):
        super().__init__()
        assert t2t_params is not None
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.t2t = t2t_params
        self.kernel_shape = t2t_params['kernel_size'][0] * t2t_params['kernel_size'][1]

    def __call__(self, x: mx.array, output_size: tuple[int, int]) -> mx.array:
        """x: (B, N, dim). N must equal T * n_vecs_h * n_vecs_w.

        Semantics from upstream:
            x = fc1(x)                # (B, N, hidden=1960)
            # fold treats the hidden axis as (C=40) * (kH*kW=49)
            # output_size is the per-frame token-grid size, e.g. (60, 108)
            # PT reshape: x.view(-1, n_vecs, hidden).permute(0,2,1) -> (B*T, hidden, n_vecs)
            # F.fold(kernel=7, stride=3, pad=3) -> (B*T, 40, H, W) overlap-sum
            # divide by F.fold(ones(B*T, 49, n_vecs)) -> (B*T, 1, H, W)
            # F.unfold(...) -> (B*T, hidden, n_vecs)
            x = fc2(x)
        """
        kernel = self.t2t['kernel_size']
        stride = self.t2t['stride']
        padding = self.t2t['padding']
        n_vecs_h, n_vecs_w = _grid_size(output_size, kernel, stride, padding)
        n_vecs = n_vecs_h * n_vecs_w

        x = self.fc1(x)  # (B, N, hidden)
        B, N, hidden = x.shape
        kk = self.kernel_shape  # k*k = 49
        assert hidden % kk == 0, f"hidden {hidden} not divisible by k*k {kk}"
        C = hidden // kk  # treat hidden as C*kk channels-x-kernel

        # ----- normalizer -----
        # ones in fold input layout: (B*T, n_vecs, kk) (C=1)
        ones = mx.ones((B * (N // n_vecs), n_vecs, kk), dtype=x.dtype)
        norm = fold_nhwc(ones, output_size, kernel, stride, padding, 1)
        norm = mx.maximum(norm, mx.array(1e-6, dtype=norm.dtype))

        # ----- main fold -----
        # x layout (PT): hidden ordering is (C, kk) inner. Our fold_nhwc now
        # also expects PT's (c, kh, kw) inner-ordering — direct reshape.
        x_re = x.reshape(B * (N // n_vecs), n_vecs, C * kk)
        folded = fold_nhwc(x_re, output_size, kernel, stride, padding, C)
        folded = folded / norm  # (B*T, H, W, C)

        # ----- main unfold -----
        unfolded = unfold_nhwc(folded, kernel, stride, padding)  # (B*T, L, C*kk)
        unfolded = unfolded.reshape(B, N, hidden)
        # upstream applies GELU before fc2 (fc2 is Sequential(GELU, Linear))
        unfolded = nn.gelu(unfolded)
        out = self.fc2(unfolded)
        return out


# ---------------------------------------------------------------------------
# Window partition helper
# ---------------------------------------------------------------------------

def window_partition(x: mx.array, window_size: tuple[int, int],
                     n_head: int) -> mx.array:
    """Mirror of upstream ``window_partition``.

    Args:
        x: (B, T, H, W, C)
    Returns:
        (B, num_w_h, num_w_w, n_head, T, win_h, win_w, C//n_head)
    """
    B, T, H, W, C = x.shape
    wH, wW = window_size
    x = x.reshape(B, T, H // wH, wH, W // wW, wW, n_head, C // n_head)
    # permute (0, 2, 4, 6, 1, 3, 5, 7)
    out = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return out


def _roll_2d(x: mx.array, sh: int, sw: int) -> mx.array:
    """Roll an NTHWC tensor along H, W axes (dims 2, 3) — matches PT torch.roll."""
    # mx has no roll; emulate via concat
    # x: (B, T, H, W, C)
    H = x.shape[2]
    W = x.shape[3]
    sh = sh % H
    sw = sw % W
    if sh:
        x = mx.concatenate([x[:, :, -sh:], x[:, :, :-sh]], axis=2)
    if sw:
        x = mx.concatenate([x[:, :, :, -sw:], x[:, :, :, :-sw]], axis=3)
    return x


# ---------------------------------------------------------------------------
# Sparse window attention
# ---------------------------------------------------------------------------

class _GroupedConvPool(nn.Module):
    """Per-channel (groups=C) Conv2d for `pool_layer`. We use nn.Conv2d
    natively since MLX supports groups>1."""
    def __init__(self, dim: int, pool_size: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=pool_size,
                              stride=pool_size, padding=(0, 0), groups=dim)


class SparseWindowAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, window_size: tuple[int, int],
                 pool_size: tuple[int, int] = (4, 4), qkv_bias: bool = True,
                 pooling_token: bool = True):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.n_head = n_head
        self.window_size = window_size
        self.pool_size = pool_size
        self.pooling_token = pooling_token
        # qkv proj
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        # output proj
        self.proj = nn.Linear(dim, dim)
        if pooling_token:
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=pool_size,
                                         stride=pool_size, padding=(0, 0),
                                         groups=dim)
        # expand_size for rolled windows
        self.expand_size = ((window_size[0] + 1) // 2, (window_size[1] + 1) // 2)
        # build valid_ind_rolled buffer (same as upstream)
        # Will be loaded from npz on weight load; default to placeholder.
        self.valid_ind_rolled = None

    def _build_valid_ind_rolled(self) -> mx.array:
        wH, wW = self.window_size
        eH, eW = self.expand_size
        import numpy as np
        def make(sl0, sl1):
            m = np.ones((wH, wW), dtype=np.float32)
            m[sl0, sl1] = 0
            return m
        # mask_tl: [:-eH, :-eW] = 0
        # mask_tr: [:-eH, eW:]  = 0
        # mask_bl: [eH:,  :-eW] = 0
        # mask_br: [eH:,  eW:]  = 0
        mtl = make(slice(0, wH - eH), slice(0, wW - eW))
        mtr = make(slice(0, wH - eH), slice(eW, wW))
        mbl = make(slice(eH, wH),     slice(0, wW - eW))
        mbr = make(slice(eH, wH),     slice(eW, wW))
        stacked = np.stack([mtl, mtr, mbl, mbr], axis=0).flatten()
        idx = np.nonzero(stacked)[0]
        return mx.array(idx.astype(np.int32))

    def __call__(self, x: mx.array, mask: mx.array,
                 T_ind: mx.array | None = None) -> mx.array:
        """x: (B, T, H, W, C). mask: (B, T, H, W, 1) — already 0/1."""
        B, T, H, W, C = x.shape
        wH, wW = self.window_size
        c_head = C // self.n_head
        n_wh = (H + wH - 1) // wH
        n_ww = (W + wW - 1) // wW
        new_H = n_wh * wH
        new_W = n_ww * wW
        pad_b = new_H - H
        pad_r = new_W - W
        if pad_b or pad_r:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_b), (0, pad_r), (0, 0)])
            mask = mx.pad(mask, [(0, 0), (0, 0), (0, pad_b), (0, pad_r), (0, 0)])

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # (B, n_wh, n_ww, n_head, T, wH, wW, c_head)
        win_q = window_partition(q, self.window_size, self.n_head)
        win_k = window_partition(k, self.window_size, self.n_head)
        win_v = window_partition(v, self.window_size, self.n_head)
        nW = n_wh * n_ww
        win_q = win_q.reshape(B, nW, self.n_head, T, wH * wW, c_head)
        win_k = win_k.reshape(B, nW, self.n_head, T, wH * wW, c_head)
        win_v = win_v.reshape(B, nW, self.n_head, T, wH * wW, c_head)

        # rolled keys/values
        eH, eW = self.expand_size
        rolled_segments_k = []
        rolled_segments_v = []
        for sh, sw in [(-eH, -eW), (-eH, eW), (eH, -eW), (eH, eW)]:
            kr = _roll_2d(k, sh, sw)
            vr = _roll_2d(v, sh, sw)
            kr_w = window_partition(kr, self.window_size, self.n_head)
            vr_w = window_partition(vr, self.window_size, self.n_head)
            kr_w = kr_w.reshape(B, nW, self.n_head, T, wH * wW, c_head)
            vr_w = vr_w.reshape(B, nW, self.n_head, T, wH * wW, c_head)
            rolled_segments_k.append(kr_w)
            rolled_segments_v.append(vr_w)
        # concat along token axis (axis=4 in upstream after the first cat)
        rool_k = mx.concatenate(rolled_segments_k, axis=4)  # (B, nW, nh, T, 4*wH*wW, c_head)
        rool_v = mx.concatenate(rolled_segments_v, axis=4)
        # gather valid_ind_rolled
        if self.valid_ind_rolled is None:
            self.valid_ind_rolled = self._build_valid_ind_rolled()
        ind = self.valid_ind_rolled.astype(mx.int32)
        rool_k = rool_k[:, :, :, :, ind, :]
        rool_v = rool_v[:, :, :, :, ind, :]
        win_k = mx.concatenate([win_k, rool_k], axis=4)
        win_v = mx.concatenate([win_v, rool_v], axis=4)

        if self.pooling_token:
            # pool_x: groupwise mean over pool_size
            # x: (B, T, new_H, new_W, C) -> reshape to (B*T, new_H, new_W, C)
            x_bt = x.reshape(B * T, new_H, new_W, C)
            pool_x = self.pool_layer(x_bt)  # (B*T, p_h, p_w, C)
            p_h = pool_x.shape[1]
            p_w = pool_x.shape[2]
            pool_x = pool_x.reshape(B, T, p_h, p_w, C)
            pool_k = self.key(pool_x)
            pool_v = self.value(pool_x)
            # repeat along nW axis
            pool_k = pool_k.reshape(B, 1, T, p_h, p_w, self.n_head, c_head)
            pool_v = pool_v.reshape(B, 1, T, p_h, p_w, self.n_head, c_head)
            pool_k = mx.broadcast_to(pool_k, (B, nW, T, p_h, p_w, self.n_head, c_head))
            pool_v = mx.broadcast_to(pool_v, (B, nW, T, p_h, p_w, self.n_head, c_head))
            # permute -> (B, nW, n_head, T, p_h, p_w, c_head)
            pool_k = pool_k.transpose(0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.transpose(0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.reshape(B, nW, self.n_head, T, p_h * p_w, c_head)
            pool_v = pool_v.reshape(B, nW, self.n_head, T, p_h * p_w, c_head)
            win_k = mx.concatenate([win_k, pool_k], axis=4)
            win_v = mx.concatenate([win_v, pool_v], axis=4)

        # mask pooling: max_pool with kernel=window_size, stride=window_size
        # mask: (B, T, new_H, new_W, 1)
        l_t = mask.shape[1]
        m = mask.reshape(B * l_t, new_H, new_W, 1)
        m = m.reshape(B * l_t, n_wh, wH, n_ww, wW, 1).transpose(0, 1, 3, 2, 4, 5)
        m = m.reshape(B * l_t, n_wh, n_ww, wH * wW)
        m = mx.max(m, axis=-1)  # (B*l_t, n_wh, n_ww)
        m = m.reshape(B, l_t, n_wh * n_ww)
        win_mask = mx.sum(m, axis=1)  # (B, nW)

        scale = 1.0 / math.sqrt(c_head)

        # We loop over batch entries (B is small at inference, usually 1)
        outs = []
        for i in range(B):
            mask_ind_arr = win_mask[i]
            # we need positions where != 0
            # MLX has no nonzero; use numpy temporarily
            mi = mx.array(mask_ind_arr.tolist())
            mask_np = mi
            # Convert via tolist trick: small array
            import numpy as np
            mask_np = np.array(mi.tolist())
            mask_ind = np.nonzero(mask_np)[0].astype(np.int32)
            unmask_ind = np.where(mask_np == 0)[0].astype(np.int32)

            out_i = mx.zeros((nW, self.n_head, T, wH * wW, c_head), dtype=x.dtype)

            if len(mask_ind):
                mind = mx.array(mask_ind)
                wq = win_q[i][mind]  # (mn, n_head, T, wH*wW, c_head)
                wk = win_k[i][mind]
                wv = win_v[i][mind]
                # apply T_ind to k, v
                if T_ind is not None:
                    Ti = T_ind  # (Ts,) int
                    wk = wk[:, :, Ti, :, :]  # (mn, n_head, Ts, wH*wW + extras, c_head)
                    wv = wv[:, :, Ti, :, :]
                    Ts = Ti.shape[0]
                else:
                    Ts = T
                mn = wq.shape[0]
                wq2 = wq.reshape(mn, self.n_head, T * wH * wW, c_head)
                wk2 = wk.reshape(mn, self.n_head, Ts * wk.shape[3], c_head)
                wv2 = wv.reshape(mn, self.n_head, Ts * wv.shape[3], c_head)
                att = (wq2 @ wk2.transpose(0, 1, 3, 2)) * scale
                att = mx.softmax(att, axis=-1)
                y = att @ wv2  # (mn, n_head, T*wH*wW, c_head)
                y = y.reshape(mn, self.n_head, T, wH * wW, c_head)
                # scatter into out_i at mask_ind
                # MLX has no scatter; build via concat-by-index. Simpler:
                # convert to numpy, scatter, convert back. But these arrays
                # can be large. We instead loop over scatter indices.
                # Trick: use mx.array advanced indexing on the LHS.
                # Pre-MLX 0.31, indexed assignment via `at` is available.
                out_i = out_i.at[mind].add(y)

            if len(unmask_ind):
                uind = mx.array(unmask_ind)
                wq = win_q[i][uind]  # (un, n_head, T, wH*wW, c_head)
                wk = win_k[i][uind, :, :, :wH * wW]  # only the per-window keys
                wv = win_v[i][uind, :, :, :wH * wW]
                # standard attention per window
                # (un, n_head, T, wH*wW, c_head) — attend within each frame
                att = (wq @ wk.transpose(0, 1, 2, 4, 3)) * scale
                att = mx.softmax(att, axis=-1)
                y = att @ wv  # (un, n_head, T, wH*wW, c_head)
                out_i = out_i.at[uind].add(y)

            outs.append(out_i)
        out = mx.stack(outs, axis=0)  # (B, nW, n_head, T, wH*wW, c_head)

        # un-window
        out = out.reshape(B, n_wh, n_ww, self.n_head, T, wH, wW, c_head)
        # permute (0, 4, 1, 5, 2, 6, 3, 7) -> (B, T, n_wh, wH, n_ww, wW, n_head, c_head)
        out = out.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        out = out.reshape(B, T, new_H, new_W, C)
        if pad_b or pad_r:
            out = out[:, :, :H, :W, :]

        out = self.proj(out)
        return out


# ---------------------------------------------------------------------------
# Transformer block & top-level
# ---------------------------------------------------------------------------

def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array,
                eps: float = 1e-5) -> mx.array:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / mx.sqrt(var + eps)
    return x * weight + bias


class _LayerNormLast(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        return _layer_norm(x, self.weight, self.bias, self.eps)


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim: int, n_head: int, window_size: tuple[int, int],
                 pool_size: tuple[int, int], t2t_params: dict):
        super().__init__()
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = _LayerNormLast(dim)
        self.norm2 = _LayerNormLast(dim)
        self.mlp = FusionFeedForward(dim, hidden_dim=1960, t2t_params=t2t_params)

    def __call__(self, x: mx.array, fold_x_size: tuple[int, int],
                 mask: mx.array, T_ind: mx.array | None = None) -> mx.array:
        B, T, H, W, C = x.shape
        shortcut = x
        xn = self.norm1(x)
        att_x = self.attention(xn, mask, T_ind)
        x = shortcut + att_x
        y = self.norm2(x)
        ff = self.mlp(y.reshape(B, T * H * W, C), fold_x_size)
        x = x + ff.reshape(B, T, H, W, C)
        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, window_size: tuple[int, int],
                 pool_size: tuple[int, int], depths: int,
                 t2t_params: dict):
        super().__init__()
        self.depths = depths
        self.blocks = [TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t_params)
                       for _ in range(depths)]

    def __call__(self, x: mx.array, fold_x_size: tuple[int, int],
                 l_mask: mx.array, t_dilation: int = 2) -> mx.array:
        T = x.shape[1]
        # Build T_ind list as upstream:
        # T_ind = [arange(i, T, t_dilation) for i in range(t_dilation)] * (depths // t_dilation)
        base = [mx.arange(i, T, t_dilation) for i in range(t_dilation)]
        T_ind_list = base * (self.depths // t_dilation)
        for i in range(self.depths):
            x = self.blocks[i](x, fold_x_size, l_mask, T_ind_list[i])
        return x

    # ---- weight loading ----
    @staticmethod
    def key_map(depths: int = 8) -> dict[str, str]:
        m: dict[str, str] = {}
        for i in range(depths):
            base = f"transformers.transformer.{i}"
            int_base = f"blocks.{i}"
            # norms
            m[f"{int_base}.norm1.weight"] = f"{base}.norm1.weight"
            m[f"{int_base}.norm1.bias"]   = f"{base}.norm1.bias"
            m[f"{int_base}.norm2.weight"] = f"{base}.norm2.weight"
            m[f"{int_base}.norm2.bias"]   = f"{base}.norm2.bias"
            # attention
            for n in ("key", "query", "value", "proj"):
                m[f"{int_base}.attention.{n}.weight"] = f"{base}.attention.{n}.weight"
                m[f"{int_base}.attention.{n}.bias"]   = f"{base}.attention.{n}.bias"
            m[f"{int_base}.attention.pool_layer.weight"] = f"{base}.attention.pool_layer.weight"
            m[f"{int_base}.attention.pool_layer.bias"]   = f"{base}.attention.pool_layer.bias"
            m[f"{int_base}.attention.valid_ind_rolled"]  = f"{base}.attention.valid_ind_rolled"
            # mlp
            m[f"{int_base}.mlp.fc1.weight"] = f"{base}.mlp.fc1.0.weight"
            m[f"{int_base}.mlp.fc1.bias"]   = f"{base}.mlp.fc1.0.bias"
            m[f"{int_base}.mlp.fc2.weight"] = f"{base}.mlp.fc2.1.weight"
            m[f"{int_base}.mlp.fc2.bias"]   = f"{base}.mlp.fc2.1.bias"
        return m

    def load_from_flat(self, flat: dict[str, mx.array]):
        m = self.key_map(self.depths)
        for internal, npz_key in m.items():
            parts = internal.split(".")
            obj = self
            for p in parts[:-1]:
                obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
            setattr(obj, parts[-1], flat[npz_key])
