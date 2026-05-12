"""FFC (Fast Fourier Convolution) blocks for LaMa, MLX port.

All modules work in NHWC internally (MLX native). FFT operates over the last two
spatial dims after a transpose to NCHW since mx.fft.rfft2 takes the last two axes.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def reflect_pad_nhwc(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad an NHWC tensor by `pad` on H and W."""
    if pad <= 0:
        return x
    # top
    top = x[:, 1 : 1 + pad, :, :][:, ::-1, :, :]
    # bottom
    bot = x[:, -1 - pad : -1, :, :][:, ::-1, :, :]
    x = mx.concatenate([top, x, bot], axis=1)
    # left
    lef = x[:, :, 1 : 1 + pad, :][:, :, ::-1, :]
    rig = x[:, :, -1 - pad : -1, :][:, :, ::-1]
    x = mx.concatenate([lef, x, rig], axis=2)
    return x


class ReflectConv2d(nn.Module):
    """Conv2d with reflect padding. Input/output NHWC."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.padding = padding
        self.dilation = dilation
        # MLX Conv2d weight: (out, kH, kW, in)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=0, dilation=dilation, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding > 0:
            # account for dilation
            eff = self.padding * self.dilation if self.dilation > 1 else self.padding
            x = reflect_pad_nhwc(x, eff)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Fourier unit
# ---------------------------------------------------------------------------

class FourierUnit(nn.Module):
    """Channel-doubling rFFT conv as in LaMa's FourierUnit."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # input: in_channels * 2 (real|imag stacked), output: out_channels * 2
        # MLX Conv2d weight: (out, kH, kW, in)
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm(out_channels * 2)
        self.relu = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        N, H, W, C = x.shape
        # Move to NCHW for FFT (FFT over last 2 axes).
        x_nchw = mx.transpose(x, (0, 3, 1, 2))
        # rfft2 over last two dims → complex, shape (N, C, H, W//2+1)
        # MLX rfft2 has no `norm` kwarg. We emulate PT norm='ortho' = 1/sqrt(N) on both
        # forward and inverse, where N = H*W.
        ffted = mx.fft.rfft2(x_nchw, axes=(-2, -1))
        inv_sqrtN = 1.0 / mx.sqrt(mx.array(float(H * W)))
        ffted = ffted * inv_sqrtN
        # Stack real/imag along a new dim, then merge with channel.
        real = ffted.real
        imag = ffted.imag
        # (N, C, 2, H, W//2+1) → (N, 2*C, H, W//2+1)
        stacked = mx.stack([real, imag], axis=2)
        N2, C2, _, Hf, Wf = stacked.shape
        merged = stacked.reshape(N2, C2 * 2, Hf, Wf)
        # Back to NHWC for MLX conv/bn
        merged_nhwc = mx.transpose(merged, (0, 2, 3, 1))
        merged_nhwc = self.conv_layer(merged_nhwc)
        merged_nhwc = self.bn(merged_nhwc)
        merged_nhwc = self.relu(merged_nhwc)
        # Back to NCHW for inverse FFT
        merged_nchw = mx.transpose(merged_nhwc, (0, 3, 1, 2))
        N3, C3, Hf2, Wf2 = merged_nchw.shape
        out_c = C3 // 2
        unstacked = merged_nchw.reshape(N3, out_c, 2, Hf2, Wf2)
        real_o = unstacked[:, :, 0, :, :]
        imag_o = unstacked[:, :, 1, :, :]
        complex_o = mx.add(real_o, mx.multiply(imag_o, mx.array(1j, dtype=mx.complex64)))
        # MLX irfft2 default divides by N = H*W. PT 'ortho' divides by sqrt(N).
        # To emulate PT ortho on the inverse side: multiply by sqrt(N) (so net 1/N → 1/sqrt(N)).
        out = mx.fft.irfft2(complex_o, s=(H, W), axes=(-2, -1))
        out = out * mx.sqrt(mx.array(float(H * W)))
        # back to NHWC
        out_nhwc = mx.transpose(out, (0, 2, 3, 1))
        return out_nhwc


# ---------------------------------------------------------------------------
# Spectral transform
# ---------------------------------------------------------------------------

class SpectralTransform(nn.Module):
    """conv1 → FourierUnit → conv2 with skip on the FFT branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 enable_lfu: bool = False):
        super().__init__()
        assert not enable_lfu, "enable_lfu=False is the LaMa big-lama config"
        assert stride == 1, "spectral_transform stride 2 not supported (not used in big-lama)"

        mid = out_channels // 2
        # conv1: Sequential(Conv2d, BN, ReLU) in PT → so PT key uses .conv1.0, .conv1.1
        self.conv1_conv = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.conv1_bn = nn.BatchNorm(mid)
        self.fu = FourierUnit(mid, mid)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        x = self.conv1_conv(x)
        x = self.conv1_bn(x)
        x = nn.relu(x)
        out_fu = self.fu(x)
        return self.conv2(x + out_fu)


# ---------------------------------------------------------------------------
# FFC core
# ---------------------------------------------------------------------------

class FFC(nn.Module):
    """4-branch FFC."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = False, enable_lfu: bool = False):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.in_cg = in_cg
        self.in_cl = in_cl
        self.out_cg = out_cg
        self.out_cl = out_cl

        # PT uses padding_mode='reflect' on every Conv2d here.
        def make_conv(in_c, out_c):
            return ReflectConv2d(in_c, out_c, kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, bias=bias)

        self.convl2l = make_conv(in_cl, out_cl) if (in_cl and out_cl) else None
        self.convl2g = make_conv(in_cl, out_cg) if (in_cl and out_cg) else None
        self.convg2l = make_conv(in_cg, out_cl) if (in_cg and out_cl) else None
        self.convg2g = (SpectralTransform(in_cg, out_cg, stride=stride, enable_lfu=enable_lfu)
                        if (in_cg and out_cg) else None)

    def __call__(self, x_l, x_g):
        # x_l/x_g: NHWC arrays or None
        out_xl = mx.zeros(()) if self.out_cl == 0 else None
        out_xg = mx.zeros(()) if self.out_cg == 0 else None

        if self.out_cl > 0:
            out_xl = None
            if self.convl2l is not None:
                out_xl = self.convl2l(x_l)
            if self.convg2l is not None:
                t = self.convg2l(x_g)
                out_xl = t if out_xl is None else out_xl + t
        if self.out_cg > 0:
            out_xg = None
            if self.convl2g is not None:
                out_xg = self.convl2g(x_l)
            if self.convg2g is not None:
                t = self.convg2g(x_g)
                out_xg = t if out_xg is None else out_xg + t
        return out_xl, out_xg


# ---------------------------------------------------------------------------
# FFC + BN + ACT wrapper
# ---------------------------------------------------------------------------

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
                 dilation: int = 1, bias: bool = False, activation: bool = True,
                 enable_lfu: bool = False):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                       stride=stride, padding=padding, dilation=dilation, bias=bias,
                       enable_lfu=enable_lfu)
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.bn_l = nn.BatchNorm(out_cl) if (ratio_gout != 1 and out_cl > 0) else None
        self.bn_g = nn.BatchNorm(out_cg) if (ratio_gout != 0 and out_cg > 0) else None
        self.do_act_l = (activation and ratio_gout != 1 and out_cl > 0)
        self.do_act_g = (activation and ratio_gout != 0 and out_cg > 0)

    def __call__(self, x_l, x_g):
        x_l, x_g = self.ffc(x_l, x_g)
        if self.bn_l is not None and x_l is not None:
            x_l = self.bn_l(x_l)
            if self.do_act_l:
                x_l = nn.relu(x_l)
        if self.bn_g is not None and x_g is not None:
            x_g = self.bn_g(x_g)
            if self.do_act_g:
                x_g = nn.relu(x_g)
        return x_l, x_g


# ---------------------------------------------------------------------------
# Resnet block of two FFC_BN_ACT
# ---------------------------------------------------------------------------

class FFCResnetBlock(nn.Module):
    def __init__(self, dim: int, ratio_gin: float, ratio_gout: float,
                 dilation: int = 1, enable_lfu: bool = False):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                padding=dilation, dilation=dilation, activation=True,
                                enable_lfu=enable_lfu)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                                padding=dilation, dilation=dilation, activation=True,
                                enable_lfu=enable_lfu)

    def __call__(self, x_l, x_g):
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1(x_l, x_g)
        x_l, x_g = self.conv2(x_l, x_g)
        if x_l is not None and id_l is not None:
            x_l = x_l + id_l
        if x_g is not None and id_g is not None:
            x_g = x_g + id_g
        return x_l, x_g
