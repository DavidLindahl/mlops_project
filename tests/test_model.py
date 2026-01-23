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
    def fake_create_model(model_name: str, pretrained: bool, num_classes: int):
        return _DummyBackbone(num_classes=num_classes)

    def fake_resolve_model_data_config(_model):
        return {"input_size": (3, 8, 8), "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    # Patch what Model() actually uses (already imported in model_mod)
    monkeypatch.setattr(model_mod.timm, "create_model", fake_create_model)
    monkeypatch.setattr(model_mod, "resolve_model_data_config", fake_resolve_model_data_config)

    m = model_mod.Model(model_name="anything", num_classes=2, pretrained=False)
    x = torch.rand(2, 3, 8, 8)
    y = m(x)
    assert y.shape == (2, 2)
