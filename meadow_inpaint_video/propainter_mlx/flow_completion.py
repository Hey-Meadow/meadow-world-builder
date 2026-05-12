"""RecurrentFlowCompletion (RFC) — MLX port.

Inference-only. Mirrors ``model.recurrent_flow_completion.RecurrentFlowCompleteNet``
from upstream ProPainter, except:
  * the EdgeDetector is only used for training, so we skip it in forward.
  * we expose the same ``forward_bidirect_flow`` and ``combine_flow`` helpers.

NCDHW tensors from PT (B, C, T, H, W) become NDHWC in MLX
(B, T, H, W, C). The PT module also accepts (B, T, C, H, W) inputs and
permutes internally; we keep the same external interface.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .raft import bilinear_sample_nhwc
from .deform_conv import modulated_deform_conv2d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _leaky_relu(x, slope=0.2):
    return mx.where(x >= 0, x, slope * x)


def _interp_bilinear_2x(x: mx.array) -> mx.array:
    """align_corners=True bilinear upsample by 2 on NHWC."""
    B, H, W, C = x.shape
    new_h = H * 2
    new_w = W * 2
    ys = mx.linspace(0, H - 1, new_h)
    xs = mx.linspace(0, W - 1, new_w)
    gy, gx = mx.meshgrid(ys, xs, indexing="ij")
    coords = mx.stack([gx, gy], axis=-1)
    coords = mx.broadcast_to(coords[None], (B, new_h, new_w, 2))
    return bilinear_sample_nhwc(x, coords)


def _pad_replicate_dhw(x: mx.array, pad: tuple) -> mx.array:
    """Pad NDHWC tensor by replicate. pad = (pT0, pT1, pH0, pH1, pW0, pW1)."""
    pT0, pT1, pH0, pH1, pW0, pW1 = pad
    if pT0:
        x = mx.concatenate(
            [mx.broadcast_to(x[:, :1], (x.shape[0], pT0, *x.shape[2:])), x], axis=1)
    if pT1:
        x = mx.concatenate(
            [x, mx.broadcast_to(x[:, -1:], (x.shape[0], pT1, *x.shape[2:]))], axis=1)
    if pH0:
        x = mx.concatenate(
            [mx.broadcast_to(x[:, :, :1], (x.shape[0], x.shape[1], pH0, *x.shape[3:])), x],
            axis=2)
    if pH1:
        x = mx.concatenate(
            [x, mx.broadcast_to(x[:, :, -1:], (x.shape[0], x.shape[1], pH1, *x.shape[3:]))],
            axis=2)
    if pW0:
        x = mx.concatenate(
            [mx.broadcast_to(x[:, :, :, :1], (*x.shape[:3], pW0, x.shape[4])), x], axis=3)
    if pW1:
        x = mx.concatenate(
            [x, mx.broadcast_to(x[:, :, :, -1:], (*x.shape[:3], pW1, x.shape[4]))], axis=3)
    return x


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

class P3DBlock(nn.Module):
    """1×k×k spatial conv followed by 3×1×1 dilated temporal conv."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch,
                                kernel_size=(1, kernel_size, kernel_size),
                                stride=(1, stride, stride),
                                padding=(0, padding, padding))
        self.conv2 = nn.Conv3d(out_ch, out_ch,
                                kernel_size=(3, 1, 1),
                                stride=(1, 1, 1),
                                padding=(2, 0, 0),
                                dilation=(2, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        f1 = _leaky_relu(self.conv1(x))
        f2 = self.conv2(f1)
        return f2


class _Deconv2d(nn.Module):
    """Bilinear 2x upsample (align_corners=True) then conv."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding)

    def __call__(self, x: mx.array) -> mx.array:
        x = _interp_bilinear_2x(x)
        return self.conv(x)


class SecondOrderDeformableAlignment(nn.Module):
    """Reproduces upstream `SecondOrderDeformableAlignment`."""
    def __init__(self, in_ch: int, out_ch: int, deform_groups: int = 16,
                 max_residue_magnitude: int = 5):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        # learnable conv weights
        self.weight = mx.zeros((out_ch, 3, 3, in_ch))
        self.bias = mx.zeros((out_ch,))
        # conv_offset: Sequential of 4 Conv2d with leakyReLU(0.1) between
        co = []
        co.append(nn.Conv2d(3 * out_ch, out_ch, 3, padding=1))
        co.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        co.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        co.append(nn.Conv2d(out_ch, 27 * deform_groups, 3, padding=1))
        self.conv_offset_0 = co[0]
        self.conv_offset_2 = co[1]
        self.conv_offset_4 = co[2]
        self.conv_offset_6 = co[3]

    def __call__(self, x: mx.array, extra_feat: mx.array) -> mx.array:
        o = self.conv_offset_0(extra_feat)
        o = _leaky_relu(o, 0.1)
        o = self.conv_offset_2(o)
        o = _leaky_relu(o, 0.1)
        o = self.conv_offset_4(o)
        o = _leaky_relu(o, 0.1)
        o = self.conv_offset_6(o)
        # chunk into 3 along channel axis
        C = o.shape[-1]
        third = C // 3
        o1 = o[..., :third]
        o2 = o[..., third:2 * third]
        mask_raw = o[..., 2 * third:]

        # offset = max_residue_magnitude * tanh(cat((o1, o2)))
        # then split back into offset_1, offset_2 (each half) and concat
        # The upstream code is essentially a no-op on the inner concat shape,
        # so the effective offset is simply max_res * tanh(cat([o1, o2])).
        offset = self.max_residue_magnitude * mx.tanh(mx.concatenate([o1, o2], axis=-1))
        mask = mx.sigmoid(mask_raw)
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            stride=1, padding=1, dilation=1, deform_groups=self.deform_groups,
        )


class BidirectionalPropagation(nn.Module):
    """Forward + backward propagation over time with deformable alignment."""
    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel
        self.deform_align_backward = SecondOrderDeformableAlignment(
            2 * channel, channel, deform_groups=16)
        self.deform_align_forward = SecondOrderDeformableAlignment(
            2 * channel, channel, deform_groups=16)
        # backbones: backward uses (2+0)*channel input, forward uses (2+1)*channel
        self.backbone_backward_0 = nn.Conv2d(2 * channel, channel, 3, padding=1)
        self.backbone_backward_2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.backbone_forward_0 = nn.Conv2d(3 * channel, channel, 3, padding=1)
        self.backbone_forward_2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.fusion = nn.Conv2d(2 * channel, channel, 1)

    def _backbone(self, name: str, x):
        if name == "backward_":
            x = self.backbone_backward_0(x)
            x = _leaky_relu(x, 0.1)
            x = self.backbone_backward_2(x)
        else:
            x = self.backbone_forward_0(x)
            x = _leaky_relu(x, 0.1)
            x = self.backbone_forward_2(x)
        return x

    def _deform(self, name: str, feat_prop, cond):
        if name == "backward_":
            return self.deform_align_backward(feat_prop, cond)
        return self.deform_align_forward(feat_prop, cond)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, H, W, C). returns same shape."""
        B, T, H, W, C = x.shape
        spatial = [x[:, i] for i in range(T)]  # list of (B, H, W, C)

        feats = {"spatial": spatial, "backward_": [None] * T, "forward_": [None] * T}

        for module_name in ("backward_", "forward_"):
            frame_idx = list(range(T))
            mapping_idx = list(range(T)) + list(range(T))[::-1]
            if "backward" in module_name:
                frame_idx = frame_idx[::-1]
            feat_prop = mx.zeros((B, H, W, self.channel), dtype=x.dtype)
            running: list[mx.array] = []
            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feat_prop
                    feat_n2 = mx.zeros_like(feat_prop)
                    cond_n2 = mx.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = running[-2]
                        cond_n2 = feat_n2
                    cond = mx.concatenate([cond_n1, feat_current, cond_n2], axis=-1)
                    fp_in = mx.concatenate([feat_prop, feat_n2], axis=-1)
                    feat_prop = self._deform(module_name, fp_in, cond)
                # gather other-stream feats
                other_streams = [k for k in ("backward_", "forward_") if k != module_name]
                feat_list = [feat_current]
                for k in other_streams:
                    if feats[k][idx] is not None:
                        feat_list.append(feats[k][idx])
                feat_list.append(feat_prop)
                feat = mx.concatenate(feat_list, axis=-1)
                feat_prop = feat_prop + self._backbone(module_name, feat)
                running.append(feat_prop)
            if "backward" in module_name:
                running = running[::-1]
            feats[module_name] = running

        outputs = []
        for i in range(T):
            align_feats = mx.concatenate(
                [feats["backward_"][i], feats["forward_"][i]], axis=-1)
            outputs.append(self.fusion(align_feats))
        return mx.stack(outputs, axis=1) + x


# ---------------------------------------------------------------------------
# RecurrentFlowCompleteNet
# ---------------------------------------------------------------------------

class RecurrentFlowCompleteNet(nn.Module):
    def __init__(self):
        super().__init__()
        # downsample: Conv3d(3, 32, kernel=(1,5,5), stride=(1,2,2), pad=(0,2,2))
        # NB: MLX Conv3d has no padding_mode='replicate'; we replicate manually.
        self.downsample_conv = nn.Conv3d(3, 32, kernel_size=(1, 5, 5),
                                          stride=(1, 2, 2), padding=(0, 0, 0))

        # encoder1
        self.enc1_b0 = P3DBlock(32, 32, 3, 1, 1)
        self.enc1_b2 = P3DBlock(32, 64, 3, 2, 1)
        # encoder2
        self.enc2_b0 = P3DBlock(64, 64, 3, 1, 1)
        self.enc2_b2 = P3DBlock(64, 128, 3, 2, 1)

        # mid_dilation: 3 Conv3d each (1,3,3), dilated
        self.mid0 = nn.Conv3d(128, 128, (1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 3, 3), dilation=(1, 3, 3))
        self.mid2 = nn.Conv3d(128, 128, (1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 2, 2), dilation=(1, 2, 2))
        self.mid4 = nn.Conv3d(128, 128, (1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), dilation=(1, 1, 1))

        # feat_prop
        self.feat_prop = BidirectionalPropagation(128)

        # decoders
        self.dec2_conv = nn.Conv2d(128, 128, 3, padding=1)
        self.dec2_deconv = _Deconv2d(128, 64, 3, 1)
        self.dec1_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.dec1_deconv = _Deconv2d(64, 32, 3, 1)
        self.upsample_conv = nn.Conv2d(32, 32, 3, padding=1)
        self.upsample_deconv = _Deconv2d(32, 2, 3, 1)

    # ---- weight loading ----------------------------------------------------
    def load_npz(self, npz_path: str | Path):
        data = np.load(str(npz_path))
        flat = {k: mx.array(data[k]) for k in data.files}
        m = self._key_map()
        missing = [v for v in m.values() if v not in flat]
        if missing:
            raise RuntimeError(f"Missing keys in npz: {missing[:6]}")
        for internal, npz_key in m.items():
            self._set(internal, flat[npz_key])
        return len(m)

    def _set(self, dotted: str, value: mx.array):
        parts = dotted.split(".")
        obj = self
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        setattr(obj, parts[-1], value)

    @staticmethod
    def _key_map() -> dict[str, str]:
        m: dict[str, str] = {}

        def add(internal, npz):
            m[internal] = npz

        add("downsample_conv.weight", "downsample.0.weight")
        add("downsample_conv.bias",   "downsample.0.bias")
        # encoder1 / encoder2 P3DBlocks
        add("enc1_b0.conv1.weight", "encoder1.0.conv1.0.weight")
        add("enc1_b0.conv1.bias",   "encoder1.0.conv1.0.bias")
        add("enc1_b0.conv2.weight", "encoder1.0.conv2.0.weight")
        add("enc1_b0.conv2.bias",   "encoder1.0.conv2.0.bias")
        add("enc1_b2.conv1.weight", "encoder1.2.conv1.0.weight")
        add("enc1_b2.conv1.bias",   "encoder1.2.conv1.0.bias")
        add("enc1_b2.conv2.weight", "encoder1.2.conv2.0.weight")
        add("enc1_b2.conv2.bias",   "encoder1.2.conv2.0.bias")
        add("enc2_b0.conv1.weight", "encoder2.0.conv1.0.weight")
        add("enc2_b0.conv1.bias",   "encoder2.0.conv1.0.bias")
        add("enc2_b0.conv2.weight", "encoder2.0.conv2.0.weight")
        add("enc2_b0.conv2.bias",   "encoder2.0.conv2.0.bias")
        add("enc2_b2.conv1.weight", "encoder2.2.conv1.0.weight")
        add("enc2_b2.conv1.bias",   "encoder2.2.conv1.0.bias")
        add("enc2_b2.conv2.weight", "encoder2.2.conv2.0.weight")
        add("enc2_b2.conv2.bias",   "encoder2.2.conv2.0.bias")
        # mid_dilation
        add("mid0.weight", "mid_dilation.0.weight")
        add("mid0.bias",   "mid_dilation.0.bias")
        add("mid2.weight", "mid_dilation.2.weight")
        add("mid2.bias",   "mid_dilation.2.bias")
        add("mid4.weight", "mid_dilation.4.weight")
        add("mid4.bias",   "mid_dilation.4.bias")
        # feat_prop deform_align
        for module, pt_name in [("feat_prop.deform_align_backward", "feat_prop_module.deform_align.backward_"),
                                 ("feat_prop.deform_align_forward",  "feat_prop_module.deform_align.forward_")]:
            add(f"{module}.weight", f"{pt_name}.weight")
            add(f"{module}.bias",   f"{pt_name}.bias")
            for i in (0, 2, 4, 6):
                add(f"{module}.conv_offset_{i}.weight", f"{pt_name}.conv_offset.{i}.weight")
                add(f"{module}.conv_offset_{i}.bias",   f"{pt_name}.conv_offset.{i}.bias")
        # backbone
        add("feat_prop.backbone_backward_0.weight", "feat_prop_module.backbone.backward_.0.weight")
        add("feat_prop.backbone_backward_0.bias",   "feat_prop_module.backbone.backward_.0.bias")
        add("feat_prop.backbone_backward_2.weight", "feat_prop_module.backbone.backward_.2.weight")
        add("feat_prop.backbone_backward_2.bias",   "feat_prop_module.backbone.backward_.2.bias")
        add("feat_prop.backbone_forward_0.weight",  "feat_prop_module.backbone.forward_.0.weight")
        add("feat_prop.backbone_forward_0.bias",    "feat_prop_module.backbone.forward_.0.bias")
        add("feat_prop.backbone_forward_2.weight",  "feat_prop_module.backbone.forward_.2.weight")
        add("feat_prop.backbone_forward_2.bias",    "feat_prop_module.backbone.forward_.2.bias")
        add("feat_prop.fusion.weight", "feat_prop_module.fusion.weight")
        add("feat_prop.fusion.bias",   "feat_prop_module.fusion.bias")
        # decoders
        add("dec2_conv.weight",   "decoder2.0.weight")
        add("dec2_conv.bias",     "decoder2.0.bias")
        add("dec2_deconv.conv.weight", "decoder2.2.conv.weight")
        add("dec2_deconv.conv.bias",   "decoder2.2.conv.bias")
        add("dec1_conv.weight",   "decoder1.0.weight")
        add("dec1_conv.bias",     "decoder1.0.bias")
        add("dec1_deconv.conv.weight", "decoder1.2.conv.weight")
        add("dec1_deconv.conv.bias",   "decoder1.2.conv.bias")
        add("upsample_conv.weight", "upsample.0.weight")
        add("upsample_conv.bias",   "upsample.0.bias")
        add("upsample_deconv.conv.weight", "upsample.2.conv.weight")
        add("upsample_deconv.conv.bias",   "upsample.2.conv.bias")
        return m

    # ---- forward -----------------------------------------------------------
    def _downsample(self, x):
        """x: (B, T, H, W, C=3). Replicate-pad H, W by 2 then Conv3d."""
        x = _pad_replicate_dhw(x, (0, 0, 2, 2, 2, 2))
        return self.downsample_conv(x)

    def forward(self, masked_flows: mx.array, masks: mx.array):
        """masked_flows: (B, T-1, 2, H, W), masks: (B, T-1, 1, H, W) — PT layout.

        Returns flow with shape (B, T-1, 2, H, W).
        """
        B, T, _, H, W = masked_flows.shape
        # build NDHWC input: (B, T, H, W, C=3)
        # PT: cat((masked_flows, masks), dim=1) on a (B, C, T, H, W) layout
        # gives (B, 3, T, H, W). Equivalent NDHWC = (B, T, H, W, 3).
        x_chans = mx.concatenate([masked_flows, masks], axis=2)  # (B, T-1, 3, H, W)
        x = x_chans.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, 3)

        x = self._downsample(x)
        x = _leaky_relu(x, 0.2)
        # encoder1
        feat_e1 = self.enc1_b0(x)
        feat_e1 = _leaky_relu(feat_e1, 0.2)
        feat_e1 = self.enc1_b2(feat_e1)
        feat_e1 = _leaky_relu(feat_e1, 0.2)
        # encoder2
        feat_e2 = self.enc2_b0(feat_e1)
        feat_e2 = _leaky_relu(feat_e2, 0.2)
        feat_e2 = self.enc2_b2(feat_e2)
        feat_e2 = _leaky_relu(feat_e2, 0.2)
        # mid_dilation
        m = self.mid0(feat_e2); m = _leaky_relu(m, 0.2)
        m = self.mid2(m);       m = _leaky_relu(m, 0.2)
        m = self.mid4(m);       m = _leaky_relu(m, 0.2)
        # feat_prop expects (B, T, H, W, C)
        feat_prop = self.feat_prop(m)
        Bt, Tt, Hf, Wf, Cf = feat_prop.shape
        feat_prop = feat_prop.reshape(Bt * Tt, Hf, Wf, Cf)

        # decoder2: residual add with feat_e1
        e1 = feat_e1.reshape(feat_e1.shape[0] * feat_e1.shape[1],
                              feat_e1.shape[2], feat_e1.shape[3], feat_e1.shape[4])
        d2 = self.dec2_conv(feat_prop)
        d2 = _leaky_relu(d2, 0.2)
        d2 = self.dec2_deconv(d2)
        d2 = _leaky_relu(d2, 0.2)
        d2 = d2 + e1

        d1 = self.dec1_conv(d2)
        d1 = _leaky_relu(d1, 0.2)
        d1 = self.dec1_deconv(d1)
        d1 = _leaky_relu(d1, 0.2)

        up = self.upsample_conv(d1)
        up = _leaky_relu(up, 0.2)
        up = self.upsample_deconv(up)  # (B*T, H, W, 2)

        flow = up.reshape(B, T, H, W, 2)
        # back to PT layout (B, T, 2, H, W)
        return flow.transpose(0, 1, 4, 2, 3)

    def forward_bidirect_flow(self, masked_flows_bi, masks):
        """masked_flows_bi: (fwd, bwd) each (B, T-1, 2, H, W);
        masks: (B, T, 1, H, W). Returns (pred_fwd, pred_bwd)."""
        masks_forward = masks[:, :-1]
        masks_backward = masks[:, 1:]
        mflow_f = masked_flows_bi[0] * (1 - masks_forward)
        mflow_b = masked_flows_bi[1] * (1 - masks_backward)
        pred_f = self.forward(mflow_f, masks_forward)
        mflow_b_flip = mflow_b[:, ::-1]
        masks_b_flip = masks_backward[:, ::-1]
        pred_b = self.forward(mflow_b_flip, masks_b_flip)
        pred_b = pred_b[:, ::-1]
        return [pred_f, pred_b]

    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1]
        masks_backward = masks[:, 1:]
        pf = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
        pb = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)
        return pf, pb
