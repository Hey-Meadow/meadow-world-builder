# LaMa MLX port — spec

## Goal

Port the **LaMa generator** (51 M params, FFC + ResNet) from PyTorch to MLX
so we can do image inpainting on Apple Silicon GPU, no torch at inference.

Inference-only. We do not port discriminator, perceptual loss, training code.

## Source

- Upstream code: `/Users/akaihuangm1/Desktop/github/lama/`
- Weights: `/Users/akaihuangm1/Desktop/github/lama-mlx/weights/big-lama/`
  - `models/best.ckpt` (391 MB, pytorch-lightning bundled)
  - `config.yaml`
- Paper: https://arxiv.org/abs/2109.07161 "Resolution-robust Large Mask Inpainting with Fourier Convolutions"

## Architecture (from upstream `FFCResNetGenerator`)

```
input: (B, 4, H, W)  -- 3 RGB + 1 mask channel (1 = remove)

block 1 (downsample 0):
  FFC(3→64, k=7, stride=1, ratio_gin=0, ratio_gout=0)    # all-local
  bn → relu

block 2 (downsample 1):
  FFC(64→128, k=3, stride=2, ratio_gin=0, ratio_gout=0)
  bn → relu

block 3 (downsample 2):
  FFC(128→256, k=3, stride=2, ratio_gin=0, ratio_gout=0)
  bn → relu

block 4 (downsample 3):
  FFC(256→512, k=3, stride=2, ratio_gin=0, ratio_gout=0.75)  # start mixing global
  bn → relu

blocks 5-22 (18 × ResBlock):
  FFC(512→512, ratio_gin=0.75, ratio_gout=0.75)
  bn → relu
  FFC(512→512, ratio_gin=0.75, ratio_gout=0.75)
  bn
  + skip

block 23 (up 0):
  ConvTranspose2d(512→256, k=3, stride=2)
  bn → relu

block 24 (up 1):
  ConvTranspose2d(256→128, k=3, stride=2)
  bn → relu

block 25 (up 2):
  ConvTranspose2d(128→64, k=3, stride=2)
  bn → relu

block 26 (head):
  Conv2d(64→3, k=7, stride=1)
  Sigmoid

output: (B, 3, H, W) RGB filled
```

## Key building block: FFC (Fast Fourier Convolution)

Each FFC has 4 conv branches (local→local, local→global, global→local, global→global):

```
x = split(input, ratio_gin) → (x_l, x_g)

out_xl = convl2l(x_l)
       + convg2l(x_g)         if ratio_gin > 0

out_xg = convl2g(x_l)         if ratio_gout > 0
       + spectral_transform(x_g)   if ratio_gin > 0 and ratio_gout > 0

return concat([out_xl, out_xg], dim=1)
```

The **spectral_transform** is the FFT trick:
- Conv1x1 → BN → ReLU
- Conv1x1 (= "fu")
- `rfft2(real input)` → concat real+imag along channel → Conv → BN → ReLU → Conv → split real/imag → `irfft2`
- The FFT branch is what makes LaMa "resolution-robust" — global receptive field via frequency-domain conv.

## MLX porting notes

- **MLX FFT**: `mlx.core.fft.rfft2` / `irfft2` exist since 0.20. Confirm shapes match torch.
- **BatchNorm**: MLX has `nn.BatchNorm`. Inference uses running mean/var (already in checkpoint).
- **ConvTranspose2d**: MLX has `nn.ConvTranspose2d`. Channels-last vs channels-first conversion needed when loading PT weights.
- **Sigmoid + clamp**: trivial.

## Weight loading

PT checkpoint has `state_dict['generator.model.{N}.{layer}.weight']` style keys.
989 generator tensors total. Plan:

1. Load PT ckpt with pytorch-lightning stubbed (we already verified this works).
2. Walk the state_dict, strip `generator.model.` prefix.
3. Map each PT key to MLX module attribute via a flat dict.
4. Transpose Conv2d weights: PT `(out, in, kH, kW)` → MLX `(out, kH, kW, in)`.
5. Transpose ConvTranspose2d weights similarly.
6. Save as a flat npz; loader attaches at construction time.

## File layout (target)

```
lama-mlx/
├── SPEC.md                          (this file)
├── README.md                        (user-facing, write last)
├── pyproject.toml
├── lama_mlx/
│   ├── __init__.py
│   ├── ffc.py                       FFC block + spectral_transform
│   ├── generator.py                 FFCResNetGenerator (top-level)
│   ├── weights.py                   PT -> MLX npz loader
│   └── inference.py                 single-image predict() helper
├── scripts/
│   ├── dump_pt_activations.py       per-block PT forward dump for parity
│   ├── convert_weights.py           ckpt -> npz
│   └── inpaint_cli.py               command-line: image + mask → output
├── tests/
│   ├── test_ffc.py                  per-FFC numerical parity vs PT
│   ├── test_generator.py            full forward parity
│   └── test_e2e.py                  end-to-end inpaint quality (PSNR vs PT)
└── weights/
    └── big-lama/                    symlink to PT weights
```

## Quality gate

- per-FFC block: max |mlx_out - pt_out| < 1e-3 fp32 (or 1e-2 bf16)
- generator full forward: max |out| diff < 5e-3 fp32
- end-to-end PSNR vs PT inpaint on LaMa_test_images: > 38 dB

## Estimated effort

- Weight inspection + dump: 0.5 day
- FFC block port + test: 1 day
- ResBlock + downsample/upsample: 1 day
- Generator assembly + end-to-end test: 0.5 day
- CLI + README: 0.5 day
- Total: ~4 days
