import csv
import random
from pathlib import Path
from types import SimpleNamespace

import mlops_project.train as train_mod
import numpy as np
import pandas as pd
import torch
from mlops_project.train import _seed_everything
from omegaconf import OmegaConf
from PIL import Image
from torch import nn


def test_seed_everything_deterministic() -> None:
    _seed_everything(123)
    a1 = random.random()
    b1 = np.random.rand(3)
    c1 = torch.rand(3)

    _seed_everything(123)
    a2 = random.random()
    b2 = np.random.rand(3)
    c2 = torch.rand(3)

    assert a1 == a2
    assert np.allclose(b1, b2)
    assert torch.allclose(c1, c2)


def _write_tiny_dataset(root: Path, n: int = 6) -> None:
    rows = []
    for i in range(n):
        name = f"img{i}.jpg"
        Image.new("RGB", (16, 16)).save(root / name)
        rows.append({"file_name": name, "label": i % 2})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


class _DummyModel(nn.Module):
    def __init__(self, model_name: str = "dummy", num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.data_config = {"input_size": (3, 16, 16), "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_smoke_train_one_epoch_tiny_data(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_dataset(data_dir, n=6)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    monkeypatch.setattr(
        train_mod,
        "HydraConfig",
        SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(run_dir)))),
    )
    monkeypatch.setattr(train_mod, "Model", _DummyModel)

    cfg = OmegaConf.create(
        {
            "data": {"data_dir": str(data_dir), "num_workers": 0},
            "model": {"model_name": "dummy", "num_classes": 2, "pretrained": False},
            "train": {
                "device": "cpu",
                "seed": 123,
                "num_epochs": 1,
                "batch_size": 2,
                "val_fraction": 0.34,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "step_size": 1,
                "gamma": 1.0,
            },
        }
    )

    train_mod.train_model(cfg)

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "checkpoints" / "last_model.pt").exists()
    assert (run_dir / "checkpoints" / "best_model.pt").exists()

    with (run_dir / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
