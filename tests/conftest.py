# tests/conftest.py
from __future__ import annotations

import sys
import types

import torch
from torch import nn


class _DummyBackbone(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.num_classes), dtype=x.dtype, device=x.device)


def pytest_configure() -> None:
    """
    Ensure imports are CI-safe by providing a minimal fake `timm` module.
    This prevents `timm -> torchvision` import issues in Ubuntu runners.
    """
    fake_timm = types.ModuleType("timm")

    def create_model(model_name: str, pretrained: bool, num_classes: int) -> nn.Module:
        _ = (model_name, pretrained)
        return _DummyBackbone(num_classes=num_classes)

    fake_timm.create_model = create_model

    fake_timm_data = types.ModuleType("timm.data")

    def resolve_model_data_config(_model: nn.Module) -> dict:
        return {"input_size": (3, 8, 8), "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    fake_timm_data.resolve_model_data_config = resolve_model_data_config

    sys.modules["timm"] = fake_timm
    sys.modules["timm.data"] = fake_timm_data
