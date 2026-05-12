"""FFCResNetGenerator MLX assembly for LaMa big-lama config.

Layout (matching PT state_dict keys generator.model.{N}.*):
  1   : FFC_BN_ACT k=7, in=4 out=64, ratio_gin=0, ratio_gout=0  (with ReflectionPad2d(3) outside)
  2   : FFC_BN_ACT k=3 stride=2 pad=1, 64→128, ratio_gout=0
  3   : FFC_BN_ACT k=3 stride=2 pad=1, 128→256, ratio_gout=0
  4   : FFC_BN_ACT k=3 stride=2 pad=1, 256→512, ratio_gin=0, ratio_gout=0.75
  5-22: FFCResnetBlock(512, ratio=0.75)  ×18
  23  : ConcatTupleLayer
  24,25,26: ConvTranspose2d(512→256), BN, ReLU
  27,28,29: ConvTranspose2d(256→128), BN, ReLU
  30,31,32: ConvTranspose2d(128→64),  BN, ReLU
  33  : ReflectionPad2d(3)
  34  : Conv2d(64→3, k=7)  → sigmoid
"""
from __future__ import annotations

from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .ffc import FFC_BN_ACT, FFCResnetBlock, reflect_pad_nhwc


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc: int = 4, output_nc: int = 3, ngf: int = 64,
                 n_downsampling: int = 3, n_blocks: int = 18,
                 ratio_g: float = 0.75, max_features: int = 1024):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        # block 1: k=7 (with reflect pad 3 applied externally)
        self.b1 = FFC_BN_ACT(input_nc, ngf, kernel_size=7, ratio_gin=0, ratio_gout=0,
                             padding=0, activation=True)

        # blocks 2..4: downsamples (n_downsampling = 3)
        downs = []
        in_ch = ngf
        for i in range(n_downsampling):
            out_ch = min(max_features, ngf * (2 ** (i + 1)))
            if i == n_downsampling - 1:
                rg_in = 0
                rg_out = ratio_g
            else:
                rg_in = 0
                rg_out = 0
            downs.append(FFC_BN_ACT(in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                                    ratio_gin=rg_in, ratio_gout=rg_out, activation=True))
            in_ch = out_ch
        self.downs = downs  # list of modules; populated as attribute below as well

        # Resnet bottleneck
        feats = in_ch  # 512
        self.res = [FFCResnetBlock(feats, ratio_gin=ratio_g, ratio_gout=ratio_g,
                                   dilation=1) for _ in range(n_blocks)]

        # Upsample: ConvTranspose2d → BN → ReLU, 3×
        ups_ct = []
        ups_bn = []
        cur = feats
        for i in range(n_downsampling):
            out_c = min(max_features, cur // 2)
            ups_ct.append(nn.ConvTranspose2d(cur, out_c, kernel_size=3, stride=2,
                                             padding=1, output_padding=1, bias=True))
            ups_bn.append(nn.BatchNorm(out_c))
            cur = out_c
        self.ups_ct = ups_ct
        self.ups_bn = ups_bn

        # final Conv2d 64→3 k=7 (with reflect pad 3 externally)
        self.final = nn.Conv2d(cur, output_nc, kernel_size=7, padding=0, bias=True)

        # Register sublists explicitly (so MLX walks them for state)
        # MLX treats list-of-modules attributes as nested.

    # ------------------------------------------------------------------
    # Forward — input NCHW (B, 4, H, W) returns NCHW (B, 3, H, W) sigmoid
    # ------------------------------------------------------------------
    def __call__(self, x_nchw: mx.array) -> mx.array:
        # NCHW → NHWC
        x = mx.transpose(x_nchw, (0, 2, 3, 1))

        # block 1: reflect pad 3 then k=7 FFC ratio_gout=0
        x = reflect_pad_nhwc(x, 3)
        x_l, x_g = self.b1(x, None)

        # downsamples
        for i, d in enumerate(self.downs):
            x_l, x_g = d(x_l, x_g)

        # resblocks (ratio_g)
        for r in self.res:
            x_l, x_g = r(x_l, x_g)

        # Concat local + global on channel axis (NHWC last dim)
        if x_g is not None:
            x = mx.concatenate([x_l, x_g], axis=-1)
        else:
            x = x_l

        # ups: ConvTranspose2d → BN → ReLU ×3
        for ct, bn in zip(self.ups_ct, self.ups_bn):
            x = ct(x)
            x = bn(x)
            x = nn.relu(x)

        # final
        x = reflect_pad_nhwc(x, 3)
        x = self.final(x)
        x = mx.sigmoid(x)

        # NHWC → NCHW
        out = mx.transpose(x, (0, 3, 1, 2))
        return out

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    @classmethod
    def from_npz(cls, npz_path: str, **kwargs):
        model = cls(**kwargs)
        loaded = np.load(npz_path)
        flat: Dict[str, mx.array] = {k: mx.array(loaded[k]) for k in loaded.files}
        # MLX path: build a flat dict matching parameter tree, then tree_unflatten.
        from mlx.utils import tree_unflatten, tree_flatten
        param_flat = dict(tree_flatten(model.parameters()))
        loaded_keys = set()
        missing = []
        bad_shape = []
        new_flat = {}
        for k in param_flat.keys():
            if k in flat:
                target_shape = param_flat[k].shape
                w = flat[k]
                if w.shape != target_shape:
                    bad_shape.append((k, w.shape, target_shape))
                    continue
                new_flat[k] = w
                loaded_keys.add(k)
            else:
                missing.append(k)
        if missing:
            print(f"[from_npz] WARN: {len(missing)} params not in npz; example: {missing[:5]}")
        if bad_shape:
            print(f"[from_npz] WARN: shape mismatch on {len(bad_shape)} params; example: {bad_shape[:3]}")
        model.update(tree_unflatten(list(new_flat.items())))
        extra = [k for k in flat.keys() if k not in loaded_keys]
        if extra:
            print(f"[from_npz] INFO: {len(extra)} npz keys not used; example: {extra[:5]}")
        # Set eval mode so BN uses running stats
        model.eval()
        return model
