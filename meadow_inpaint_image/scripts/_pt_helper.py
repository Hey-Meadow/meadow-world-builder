"""Helper to import the upstream PyTorch LaMa generator without needing pytorch_lightning."""
from __future__ import annotations

import sys
import types
from pathlib import Path


def install_pl_stubs():
    def _stub(n):
        m = types.ModuleType(n)
        m.__path__ = []
        sys.modules[n] = m
        return m

    class _G:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)

    if "pytorch_lightning" in sys.modules:
        return

    pl = _stub("pytorch_lightning")
    pl.seed_everything = lambda *a, **k: None
    pl.LightningModule = _G
    pl.LightningDataModule = _G
    pl.Trainer = _G
    _stub("pytorch_lightning.callbacks")
    cb = _stub("pytorch_lightning.callbacks.model_checkpoint")
    cb.ModelCheckpoint = _G
    _stub("pytorch_lightning.utilities")
    _stub("pytorch_lightning.utilities.cloud_io")
    _stub("pytorch_lightning.utilities.distributed")
    _stub("pytorch_lightning.core")
    cl = _stub("pytorch_lightning.core.lightning")
    cl.LightningModule = _G
    _stub("pytorch_lightning.loggers")


def load_pt_generator(ckpt_path: str | Path):
    install_pl_stubs()
    upstream = Path("/Users/akaihuangm1/Desktop/github/lama")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    import torch
    from saicinpainting.training.modules.ffc import FFCResNetGenerator

    net = FFCResNetGenerator(
        input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18,
        add_out_act="sigmoid",
        init_conv_kwargs=dict(ratio_gin=0, ratio_gout=0, enable_lfu=False),
        downsample_conv_kwargs=dict(ratio_gin=0, ratio_gout=0, enable_lfu=False),
        resnet_conv_kwargs=dict(ratio_gin=0.75, ratio_gout=0.75, enable_lfu=False),
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    gen_sd = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    miss, unex = net.load_state_dict(gen_sd, strict=False)
    if miss or unex:
        print(f"[pt] missing={len(miss)} unexpected={len(unex)}")
    net.eval()
    return net
