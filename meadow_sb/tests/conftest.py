"""Shared pytest fixtures for meadow_sb tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

CKPT = Path(__file__).resolve().parents[2] / "research" / "yonosplat_bootstrap" / \
       "weights" / "yonosplat" / "re10k_224x224_ctx2to32.ckpt"


@pytest.fixture(scope="session")
def sd():
    """Full state_dict from the re10k YoNoSplat checkpoint."""
    if not CKPT.exists():
        pytest.skip(f"checkpoint not present at {CKPT}")
    return torch.load(str(CKPT), map_location="cpu", weights_only=False)["state_dict"]


@pytest.fixture(scope="session")
def dumps_dir():
    return Path(__file__).resolve().parents[2] / "research" / "yonosplat_bootstrap" / "dumps"
