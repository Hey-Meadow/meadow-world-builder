"""ProPainter ``feat_prop_module`` — MLX port.

Mirrors ``model.propainter.BidirectionalPropagation`` with ``learnable=True``
(channel=128) and the per-frame ``DeformableAlignment`` (which differs from
the RFC's ``SecondOrderDeformableAlignment`` in how ``conv_offset`` is
shaped and in that the predicted offset is added to a flipped+repeated
``flow_prop``).

This also exports ``img_prop`` — the ``learnable=False`` variant used for
the image-level propagation step before the main inpainter forward.

Conventions: NHWC throughout. Flow tensors carry shape ``(B, T-1, 2, H, W)``
in PT layout for external interfaces but are converted to ``(B, T-1, H, W, 2)``
internally before sampling.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

from .raft import bilinear_sample_nhwc
from .deform_conv import modulated_deform_conv2d


def _leaky_relu(x, slope=0.2):
    return mx.where(x >= 0, x, slope * x)


def flow_warp(x: mx.array, flow: mx.array, interpolation: str = "bilinear") -> mx.array:
    """Warp ``x`` (B, H, W, C) by ``flow`` (B, H, W, 2) in pixel units.

    Equivalent to PT ``F.grid_sample(x, scaled_flow, mode=interpolation,
    padding_mode='zeros', align_corners=True)`` where the scaled flow is
    ``(grid + flow) / ((W-1)/2, (H-1)/2) - 1``.
    """
    B, H, W, C = x.shape
    # grid_x, grid_y in pixel coords (align_corners semantics)
    ys = mx.arange(H).astype(mx.float32)
    xs = mx.arange(W).astype(mx.float32)
    gy, gx = mx.meshgrid(ys, xs, indexing="ij")  # (H, W)
    grid = mx.stack([gx, gy], axis=-1)           # (H, W, 2)
    coords = grid[None] + flow                   # (B, H, W, 2) in pixel units
    if interpolation == "nearest":
        # nearest-neighbour sampling — round to int and clamp
        gx_c = mx.clip(mx.round(coords[..., 0]), 0, W - 1).astype(mx.int32)
        gy_c = mx.clip(mx.round(coords[..., 1]), 0, H - 1).astype(mx.int32)
        b_idx = mx.broadcast_to(mx.arange(B).reshape(B, 1, 1), (B, H, W))
        out = x[b_idx, gy_c, gx_c, :]
        # zero out OOB samples
        valid = (
            (coords[..., 0] >= 0) & (coords[..., 0] <= W - 1) &
            (coords[..., 1] >= 0) & (coords[..., 1] <= H - 1)
        )
        out = out * valid.astype(out.dtype)[..., None]
        return out
    return bilinear_sample_nhwc(x, coords)


def _length_sq(x_chw: mx.array) -> mx.array:
    """Sum of squares along channel axis of (B, H, W, C). Keeps channel dim."""
    return mx.sum(x_chw * x_chw, axis=-1, keepdims=True)


def fb_consistency_check(flow_fw: mx.array, flow_bw: mx.array,
                          alpha1: float = 0.01, alpha2: float = 0.5) -> mx.array:
    """Mirror of ``model.propainter.fbConsistencyCheck``.

    Args:
        flow_fw, flow_bw: (B, H, W, 2) — pixel-unit flow.
    Returns:
        valid_fw: (B, H, W, 1) float — 1 where flows are consistent.
    """
    flow_bw_warped = flow_warp(flow_bw, flow_fw)
    flow_diff_fw = flow_fw + flow_bw_warped
    mag_sq_fw = _length_sq(flow_fw) + _length_sq(flow_bw_warped)
    occ_thresh = alpha1 * mag_sq_fw + alpha2
    valid_fw = (_length_sq(flow_diff_fw) < occ_thresh).astype(mx.float32)
    return valid_fw


# ---------------------------------------------------------------------------
# DeformableAlignment for feat_prop (channel=128, deform_groups=16)
# ---------------------------------------------------------------------------

class DeformableAlignment(nn.Module):
    """Per-frame deformable alignment used in feat_prop_module."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, deform_groups: int = 16,
                 max_residue_magnitude: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        # conv weight (Cout, kH, kW, Cin)
        self.weight = mx.zeros((out_channels, kernel_size, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))
        # conv_offset: 4 Conv2d in a Sequential with LeakyReLU(0.1) between
        # input channels = 2*out_channels + 2 + 1 + 2 = 2C + 5
        in_off = 2 * out_channels + 2 + 1 + 2
        self.conv_offset_0 = nn.Conv2d(in_off, out_channels, 3, stride=1, padding=1)
        self.conv_offset_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_offset_4 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.conv_offset_6 = nn.Conv2d(out_channels, 27 * deform_groups, 3, stride=1, padding=1)

    def __call__(self, x: mx.array, cond_feat: mx.array, flow: mx.array) -> mx.array:
        """x: (B, H, W, C)  cond_feat: (B, H, W, 2C+5)  flow: (B, H, W, 2).
        Returns: (B, H, W, C)."""
        o = self.conv_offset_0(cond_feat)
        o = _leaky_relu(o, 0.1)
        o = self.conv_offset_2(o)
        o = _leaky_relu(o, 0.1)
        o = self.conv_offset_4(o)
        o = _leaky_relu(o, 0.1)
        o = self.conv_offset_6(o)
        # chunk into 3 along last axis (PT does dim=1 on NCHW = our last dim)
        C = o.shape[-1]
        third = C // 3
        o1 = o[..., :third]
        o2 = o[..., third:2 * third]
        mask_raw = o[..., 2 * third:]

        offset = self.max_residue_magnitude * mx.tanh(mx.concatenate([o1, o2], axis=-1))
        # offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        # PT 'flip(1)' flips channel axis (only 2 channels) so (fx, fy) -> (fy, fx).
        # Then repeat along channel axis offset.size(1)//2 times to match offset.
        flow_flipped = mx.concatenate([flow[..., 1:2], flow[..., 0:1]], axis=-1)
        n_rep = offset.shape[-1] // 2
        flow_rep = mx.broadcast_to(flow_flipped[..., None, :],
                                   (*flow_flipped.shape[:-1], n_rep, 2))
        flow_rep = flow_rep.reshape(*flow_flipped.shape[:-1], n_rep * 2)
        offset = offset + flow_rep

        mask = mx.sigmoid(mask_raw)
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            stride=1, padding=self.padding, dilation=1,
            deform_groups=self.deform_groups,
        )


# ---------------------------------------------------------------------------
# BidirectionalPropagation (learnable=True, channel=128)
# ---------------------------------------------------------------------------

class BidirectionalPropagation(nn.Module):
    def __init__(self, channel: int, learnable: bool = True):
        super().__init__()
        self.channel = channel
        self.learnable = learnable
        self.prop_list = ("backward_1", "forward_1")
        if learnable:
            self.deform_align_backward_1 = DeformableAlignment(channel, channel, 3, padding=1, deform_groups=16)
            self.deform_align_forward_1  = DeformableAlignment(channel, channel, 3, padding=1, deform_groups=16)
            self.backbone_backward_1_0 = nn.Conv2d(2 * channel + 2, channel, 3, stride=1, padding=1)
            self.backbone_backward_1_2 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
            self.backbone_forward_1_0  = nn.Conv2d(2 * channel + 2, channel, 3, stride=1, padding=1)
            self.backbone_forward_1_2  = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
            self.fuse_0 = nn.Conv2d(2 * channel + 2, channel, 3, stride=1, padding=1)
            self.fuse_2 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)

    def _deform(self, name: str, feat_prop, cond, flow):
        if name == "backward_1":
            return self.deform_align_backward_1(feat_prop, cond, flow)
        return self.deform_align_forward_1(feat_prop, cond, flow)

    def _backbone(self, name: str, x):
        if name == "backward_1":
            x = self.backbone_backward_1_0(x); x = _leaky_relu(x, 0.2)
            x = self.backbone_backward_1_2(x)
        else:
            x = self.backbone_forward_1_0(x); x = _leaky_relu(x, 0.2)
            x = self.backbone_forward_1_2(x)
        return x

    @staticmethod
    def _binary(mask, th: float = 0.1):
        return (mask > th).astype(mask.dtype)

    def __call__(self, x: mx.array, flows_forward: mx.array, flows_backward: mx.array,
                 mask: mx.array, interpolation: str = "bilinear"):
        """x: (B, T, H, W, C). flows_forward, flows_backward: (B, T-1, H, W, 2).
        mask: (B, T, H, W, mask_c).  Returns same-shape outputs.

        Returns:
            outputs_b, outputs_f, outputs, masks_f
        """
        B, T, H, W, C = x.shape
        feats = {"input": [x[:, i] for i in range(T)],
                  "backward_1": [None] * T, "forward_1": [None] * T}
        masks = {"input": [mask[:, i] for i in range(T)],
                  "backward_1": [None] * T, "forward_1": [None] * T}
        cache_list = ["input"] + list(self.prop_list)

        for p_i, module_name in enumerate(self.prop_list):
            feats[module_name] = []
            masks[module_name] = []
            if "backward" in module_name:
                frame_idx = list(range(T))[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = list(range(T))
                flow_idx = [-1] + list(range(T - 1))  # range(-1, T-1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop  = flows_for_prop[:, flow_idx[i]]   # (B, H, W, 2)
                    flow_check = flows_for_check[:, flow_idx[i]]
                    flow_valid = fb_consistency_check(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop, interpolation)
                    if self.learnable:
                        cond = mx.concatenate([feat_current, feat_warped, flow_prop,
                                                flow_valid, mask_current], axis=-1)
                        feat_prop = self._deform(module_name, feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop)
                        mask_prop_valid = self._binary(mask_prop_valid)
                        union = self._binary(mask_current * flow_valid * (1 - mask_prop_valid))
                        feat_prop = union * feat_warped + (1 - union) * feat_current
                        mask_prop = self._binary(mask_current * (1 - (flow_valid * (1 - mask_prop_valid))))

                if self.learnable:
                    feat = mx.concatenate([feat_current, feat_prop, mask_current], axis=-1)
                    feat_prop = feat_prop + self._backbone(module_name, feat)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        # stack
        outputs_b = mx.stack(feats["backward_1"], axis=1)  # (B, T, H, W, C)
        outputs_f = mx.stack(feats["forward_1"], axis=1)

        if self.learnable:
            # mask_in: (B*T, H, W, mask_c)
            mc = mask.shape[-1]
            mask_in = mask.reshape(B * T, H, W, mc)
            ob = outputs_b.reshape(B * T, H, W, C)
            of = outputs_f.reshape(B * T, H, W, C)
            fused_in = mx.concatenate([ob, of, mask_in], axis=-1)
            fused = self.fuse_0(fused_in); fused = _leaky_relu(fused, 0.2)
            fused = self.fuse_2(fused)
            outputs = fused.reshape(B, T, H, W, C) + x
            masks_f = None
        else:
            outputs = outputs_f
            masks_f = mx.stack(masks["forward_1"], axis=1)

        return outputs_b, outputs_f, outputs, masks_f

    # ---- weight loading ----------------------------------------------------
    @staticmethod
    def key_map(prefix: str = "feat_prop_module.") -> dict[str, str]:
        m: dict[str, str] = {}
        for direction in ("backward_1", "forward_1"):
            base = f"{prefix}deform_align.{direction}"
            int_base = f"deform_align_{direction}"
            m[f"{int_base}.weight"] = f"{base}.weight"
            m[f"{int_base}.bias"]   = f"{base}.bias"
            for i in (0, 2, 4, 6):
                m[f"{int_base}.conv_offset_{i}.weight"] = f"{base}.conv_offset.{i}.weight"
                m[f"{int_base}.conv_offset_{i}.bias"]   = f"{base}.conv_offset.{i}.bias"
            base2 = f"{prefix}backbone.{direction}"
            m[f"backbone_{direction}_0.weight"] = f"{base2}.0.weight"
            m[f"backbone_{direction}_0.bias"]   = f"{base2}.0.bias"
            m[f"backbone_{direction}_2.weight"] = f"{base2}.2.weight"
            m[f"backbone_{direction}_2.bias"]   = f"{base2}.2.bias"
        m["fuse_0.weight"] = f"{prefix}fuse.0.weight"
        m["fuse_0.bias"]   = f"{prefix}fuse.0.bias"
        m["fuse_2.weight"] = f"{prefix}fuse.2.weight"
        m["fuse_2.bias"]   = f"{prefix}fuse.2.bias"
        return m

    def load_from_flat(self, flat: dict[str, mx.array], prefix: str = "feat_prop_module."):
        m = self.key_map(prefix)
        for internal, key in m.items():
            parts = internal.split(".")
            obj = self
            for p in parts[:-1]:
                obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
            setattr(obj, parts[-1], flat[key])
