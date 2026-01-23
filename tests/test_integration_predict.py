from pathlib import Path

import torch
from torch import nn


class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_load_model_and_predict(tmp_path: Path) -> None:
    model = _TinyNet(num_classes=2)

    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    loaded = _TinyNet(num_classes=2)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    loaded.load_state_dict(ckpt["state_dict"])
    loaded.eval()

    x = torch.randn(3, 4)
    with torch.no_grad():
        logits = loaded(x)

    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
