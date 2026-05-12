"""ProPainter main-network Decoder — MLX port.

Mirrors ``model.propainter.InpaintGenerator.decoder`` (which is an
nn.Sequential containing two ``deconv`` blocks interleaved with
3x3 Conv2d layers + LeakyReLU(0.2)).
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

from .flow_completion import _leaky_relu, _interp_bilinear_2x


class _Deconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=padding)

    def __call__(self, x):
        x = _interp_bilinear_2x(x)
        return self.conv(x)


class Decoder(nn.Module):
    """128 -> 3 RGB, 4x spatial upsample, NHWC."""
    def __init__(self):
        super().__init__()
        self.d0 = _Deconv(128, 128, 3, 1)
        self.d2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.d4 = _Deconv(64, 64, 3, 1)
        self.d6 = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def __call__(self, x: mx.array, apply_tanh: bool = False) -> mx.array:
        x = self.d0(x); x = _leaky_relu(x, 0.2)
        x = self.d2(x); x = _leaky_relu(x, 0.2)
        x = self.d4(x); x = _leaky_relu(x, 0.2)
        x = self.d6(x)
        if apply_tanh:
            x = mx.tanh(x)
        return x

    @staticmethod
    def key_map() -> dict[str, str]:
        # upstream: decoder.{0,2,4,6} — index 0 and 4 are deconv (.conv.weight)
        return {
            "d0.conv.weight": "decoder.0.conv.weight",
            "d0.conv.bias":   "decoder.0.conv.bias",
            "d2.weight":      "decoder.2.weight",
            "d2.bias":        "decoder.2.bias",
            "d4.conv.weight": "decoder.4.conv.weight",
            "d4.conv.bias":   "decoder.4.conv.bias",
            "d6.weight":      "decoder.6.weight",
            "d6.bias":        "decoder.6.bias",
        }

    def load_from_flat(self, flat, prefix: str = ""):
        for internal, npz_key in self.key_map().items():
            parts = internal.split(".")
            obj = self
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], flat[prefix + npz_key])
