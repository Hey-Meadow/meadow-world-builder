"""Quality-gate tests for `meadow_sb.models.heads`.

For each of the 5 heads we:
    1. Pull the relevant weight slice from the YoNoSplat re10k checkpoint.
    2. Build a tiny PyTorch reference module with identical weights.
    3. Run the same random input through both PT and MLX.
    4. Assert max abs diff < 1e-4 (linear heads should be much tighter; the
       CameraHead has a couple of ReLUs but still well within 1e-4).

Run:
    /Users/akaihuangm1/Desktop/github/sam-3d-body/.venv/bin/python \
        -m meadow_sb.tests.test_heads
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT = REPO_ROOT / "research" / "yonosplat_bootstrap" / "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"

sys.path.insert(0, str(REPO_ROOT))

from meadow_sb.models.heads import (  # noqa: E402
    GaussianHead,
    PointHead,
    CameraHead,
    IntrinsicHead,
    RgbEmbed,
    load_gaussian_head,
    load_point_head,
    load_camera_head,
    load_intrinsic_head,
    load_rgb_embed,
    rgb_embed_from_pt,
)


# ---------------------------------------------------------------------------
# PT reference modules (faithful copies of upstream, minus the dispatch glue)


class _PTResConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.res_conv1 = nn.Linear(dim, dim)
        self.res_conv2 = nn.Linear(dim, dim)
        self.res_conv3 = nn.Linear(dim, dim)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        return res + x


class _PTCameraHead(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.res_conv = nn.ModuleList([_PTResConvBlock(dim), _PTResConvBlock(dim)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_rot = nn.Linear(dim, 9)

    def forward(self, feat, patch_h, patch_w):
        BN, hw, c = feat.shape
        for i in range(2):
            feat = self.res_conv[i](feat)
        feat = self.avgpool(
            feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous()
        )
        feat = feat.view(feat.size(0), -1)
        feat = self.more_mlps(feat)
        out_t = self.fc_t(feat)
        out_r = self.fc_rot(feat)
        return torch.cat([out_r, out_t], dim=-1)  # (BN, 12)


class _PTRgbEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=2048, patch_size=7):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)              # (B, C, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Driver


def _maxdiff(a_mx: mx.array, b_np: np.ndarray) -> float:
    a = np.asarray(a_mx)
    return float(np.max(np.abs(a - b_np)))


def _load_state_dict():
    print(f"[heads-test] loading checkpoint {CKPT}")
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    return ckpt["state_dict"]


def test_gaussian_head(sd):
    print("\n[heads-test] GaussianHead")
    w = sd["encoder.gaussian_head.proj.weight"].float()
    b = sd["encoder.gaussian_head.proj.bias"].float()
    pt = nn.Linear(1024, 539)
    with torch.no_grad():
        pt.weight.copy_(w)
        pt.bias.copy_(b)
    mlx = load_gaussian_head(sd)

    torch.manual_seed(0)
    x_pt = torch.randn(2, 16, 1024)
    y_pt = pt(x_pt).detach().numpy()
    y_mx = mlx(mx.array(x_pt.numpy()))
    d = _maxdiff(y_mx, y_pt)
    n_params = w.numel() + b.numel()
    print(f"  input={tuple(x_pt.shape)} output={tuple(y_pt.shape)} params={n_params:,}")
    print(f"  max abs diff = {d:.3e}")
    assert d < 1e-4, f"gaussian_head diff {d} >= 1e-4"
    return d


def test_point_head(sd):
    print("\n[heads-test] PointHead")
    w = sd["encoder.point_head.proj.weight"].float()
    b = sd["encoder.point_head.proj.bias"].float()
    out_dim = w.shape[0]
    pt = nn.Linear(1024, out_dim)
    with torch.no_grad():
        pt.weight.copy_(w)
        pt.bias.copy_(b)
    mlx = load_point_head(sd)

    torch.manual_seed(1)
    x_pt = torch.randn(2, 16, 1024)
    y_pt = pt(x_pt).detach().numpy()
    y_mx = mlx(mx.array(x_pt.numpy()))
    d = _maxdiff(y_mx, y_pt)
    n_params = w.numel() + b.numel()
    print(f"  input={tuple(x_pt.shape)} output={tuple(y_pt.shape)} params={n_params:,}")
    print(f"  max abs diff = {d:.3e}")
    assert d < 1e-4, f"point_head diff {d} >= 1e-4"
    return d


def test_camera_head(sd):
    print("\n[heads-test] CameraHead")
    pt = _PTCameraHead(dim=512)
    pt_sd = pt.state_dict()
    # Fill from checkpoint
    for i in range(2):
        for j in (1, 2, 3):
            pt_sd[f"res_conv.{i}.res_conv{j}.weight"].copy_(sd[f"encoder.camera_head.res_conv.{i}.res_conv{j}.weight"])
            pt_sd[f"res_conv.{i}.res_conv{j}.bias"].copy_(sd[f"encoder.camera_head.res_conv.{i}.res_conv{j}.bias"])
    pt_sd["more_mlps.0.weight"].copy_(sd["encoder.camera_head.more_mlps.0.weight"])
    pt_sd["more_mlps.0.bias"].copy_(sd["encoder.camera_head.more_mlps.0.bias"])
    pt_sd["more_mlps.2.weight"].copy_(sd["encoder.camera_head.more_mlps.2.weight"])
    pt_sd["more_mlps.2.bias"].copy_(sd["encoder.camera_head.more_mlps.2.bias"])
    pt_sd["fc_t.weight"].copy_(sd["encoder.camera_head.fc_t.weight"])
    pt_sd["fc_t.bias"].copy_(sd["encoder.camera_head.fc_t.bias"])
    pt_sd["fc_rot.weight"].copy_(sd["encoder.camera_head.fc_rot.weight"])
    pt_sd["fc_rot.bias"].copy_(sd["encoder.camera_head.fc_rot.bias"])
    pt.load_state_dict(pt_sd)
    pt.eval()

    mlx = load_camera_head(sd)

    # 224/14 = 16 patches per axis on the camera_decoder stream.
    patch_h, patch_w = 16, 16
    BN = 2
    torch.manual_seed(2)
    x_pt = torch.randn(BN, patch_h * patch_w, 512)
    with torch.no_grad():
        y_pt = pt(x_pt, patch_h, patch_w).detach().numpy()
    y_mx = mlx(mx.array(x_pt.numpy()), patch_h, patch_w)
    d = _maxdiff(y_mx, y_pt)
    n_params = sum(p.numel() for p in pt.parameters())
    print(f"  input={tuple(x_pt.shape)} output={tuple(y_pt.shape)} params={n_params:,}")
    print(f"  max abs diff = {d:.3e}")
    assert d < 1e-4, f"camera_head diff {d} >= 1e-4"
    return d


def test_intrinsic_head(sd):
    print("\n[heads-test] IntrinsicHead")
    pt = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2),
    )
    with torch.no_grad():
        pt[0].weight.copy_(sd["encoder.backbone.intrinsic_head.fc1.weight"])
        pt[0].bias.copy_(sd["encoder.backbone.intrinsic_head.fc1.bias"])
        pt[2].weight.copy_(sd["encoder.backbone.intrinsic_head.fc2.weight"])
        pt[2].bias.copy_(sd["encoder.backbone.intrinsic_head.fc2.bias"])
    pt.eval()
    mlx = load_intrinsic_head(sd)

    torch.manual_seed(3)
    x_pt = torch.randn(4, 1024)
    with torch.no_grad():
        y_pt = pt(x_pt).detach().numpy()
    y_mx = mlx(mx.array(x_pt.numpy()))
    d = _maxdiff(y_mx, y_pt)
    n_params = sum(p.numel() for p in pt.parameters())
    print(f"  input={tuple(x_pt.shape)} output={tuple(y_pt.shape)} params={n_params:,}")
    print(f"  max abs diff = {d:.3e}")
    assert d < 1e-4, f"intrinsic_head diff {d} >= 1e-4"
    return d


def test_rgb_embed(sd):
    print("\n[heads-test] RgbEmbed")
    pt = _PTRgbEmbed(in_chans=3, embed_dim=2048, patch_size=7)
    with torch.no_grad():
        pt.proj.weight.copy_(sd["encoder.rgb_embed.proj.weight"])
        pt.proj.bias.copy_(sd["encoder.rgb_embed.proj.bias"])
        pt.norm.weight.copy_(sd["encoder.rgb_embed.norm.weight"])
        pt.norm.bias.copy_(sd["encoder.rgb_embed.norm.bias"])
    pt.eval()
    mlx = load_rgb_embed(sd)

    torch.manual_seed(4)
    x_pt = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y_pt = pt(x_pt).detach().numpy()
    x_mx = rgb_embed_from_pt(mx.array(x_pt.numpy()))
    y_mx = mlx(x_mx)
    d = _maxdiff(y_mx, y_pt)
    n_params = sum(p.numel() for p in pt.parameters())
    print(f"  input={tuple(x_pt.shape)} output={tuple(y_pt.shape)} params={n_params:,}")
    print(f"  max abs diff = {d:.3e}")
    # Conv2d with 2048 channels accumulates more rounding; loosen slightly
    # but keep within the contract's 1e-4 fp32 target.
    assert d < 1e-4, f"rgb_embed diff {d} >= 1e-4"
    return d


def main():
    sd = _load_state_dict()
    diffs = {
        "gaussian_head": test_gaussian_head(sd),
        "point_head": test_point_head(sd),
        "camera_head": test_camera_head(sd),
        "intrinsic_head": test_intrinsic_head(sd),
        "rgb_embed": test_rgb_embed(sd),
    }
    print("\n[heads-test] summary")
    for k, v in diffs.items():
        print(f"  {k:16s}  max|Δ| = {v:.3e}")
    print("[heads-test] all heads PASS  (max abs diff < 1e-4)")


if __name__ == "__main__":
    main()
