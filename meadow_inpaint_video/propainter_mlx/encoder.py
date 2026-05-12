"""ProPainter main-network Encoder — MLX port.

Mirrors ``model.propainter.Encoder``. The encoder takes the concatenation of
(frame, original_mask, updated_mask) -> 5 channels and produces 128-channel
features at 1/4 spatial resolution.

Architecture: a U-shaped stack of Conv2d -> LeakyReLU(0.2) with grouped
convolutions and 4 skip connections that progressively halve channels.

NHWC convention.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .flow_completion import _leaky_relu


# the original Encoder uses grouped conv2d at later layers. MLX's nn.Conv2d
# supports groups>1 if Cin and Cout divide evenly.

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 -> 64 stride=2
        self.c0 = nn.Conv2d(5, 64, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.c4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.c6 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.c8 = nn.Conv2d(256, 384, 3, stride=1, padding=1, groups=1)
        # the skip is the output at index 8 (x0). From now on every even-index
        # conv has input doubled by skip-concat with x0 reshaped.
        self.c10 = nn.Conv2d(640, 512, 3, stride=1, padding=1, groups=2)
        self.c12 = nn.Conv2d(768, 384, 3, stride=1, padding=1, groups=4)
        self.c14 = nn.Conv2d(640, 256, 3, stride=1, padding=1, groups=8)
        self.c16 = nn.Conv2d(512, 128, 3, stride=1, padding=1, groups=1)
        self._group = [1, 2, 4, 8, 1]

    def __call__(self, x: mx.array) -> mx.array:
        """x: (BT, H, W, 5). Returns (BT, H/4, W/4, 128)."""
        # we'll mimic upstream's loop semantics exactly
        layers = [self.c0, self.c2, self.c4, self.c6, self.c8,
                  self.c10, self.c12, self.c14, self.c16]
        # Indices in upstream (sequential) are 0..17 alternating conv/leakyrelu.
        # Upstream behavior: at i=8 it captures x0 = out; for i in {10,12,14,16}
        # before applying conv it concatenates x0 (reshaped into groups) with
        # current out (reshaped into groups).
        out = x
        x0 = None
        h = w = None
        for li, layer in enumerate(layers):
            i = li * 2  # upstream sequential index for this conv
            if i == 8:
                x0 = out
                _, h, w, _ = x0.shape
            if i > 8:  # i in {10, 12, 14, 16}
                g = self._group[(i - 8) // 2]
                bt = out.shape[0]
                # NHWC -> (bt, H, W, g, c/g)
                x_sk = x0.reshape(bt, h, w, g, -1)
                o = out.reshape(bt, h, w, g, -1)
                out = mx.concatenate([x_sk, o], axis=-1).reshape(bt, h, w, -1)
            out = layer(out)
            out = _leaky_relu(out, 0.2)
        return out

    # ---- weight loading ----------------------------------------------
    @staticmethod
    def key_map() -> dict[str, str]:
        m = {}
        # upstream is nn.ModuleList: encoder.layers.{i}.weight where i is
        # the conv layer index in {0, 2, 4, 6, 8, 10, 12, 14, 16}
        for li, idx in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16]):
            m[f"c{idx}.weight"] = f"encoder.layers.{idx}.weight"
            m[f"c{idx}.bias"]   = f"encoder.layers.{idx}.bias"
        return m

    def load_from_flat(self, flat: dict[str, mx.array], prefix: str = ""):
        m = self.key_map()
        for internal, npz_key in m.items():
            key = prefix + npz_key
            parts = internal.split(".")
            obj = self
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], flat[key])
