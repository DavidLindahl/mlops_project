import sys
import types

import mlops_project.model as model_mod
import torch
from torch import nn


class _DummyBackbone(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.num_classes), dtype=x.dtype, device=x.device)


def test_model_forward_shape_offline(monkeypatch) -> None:
    fake_timm = types.SimpleNamespace()

    def create_model(model_name, pretrained, num_classes):
        return _DummyBackbone(num_classes=num_classes)

    fake_timm.create_model = create_model

    fake_timm_data = types.SimpleNamespace()

    def resolve_model_data_config(_model):
        return {"input_size": (3, 8, 8), "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    fake_timm_data.resolve_model_data_config = resolve_model_data_config

    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setitem(sys.modules, "timm.data", fake_timm_data)

    m = model_mod.Model(model_name="anything", num_classes=2, pretrained=False)
    x = torch.rand(2, 3, 8, 8)
    y = m(x)
    assert y.shape == (2, 2)
