from __future__ import annotations

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


def _write_split(root: Path, split: str, n: int) -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        name = f"{split}_img{i}.jpg"
        Image.new("RGB", (16, 16)).save(split_dir / name)
        rows.append({"file_name": name, "label": i % 2})

    csv_path = root / f"{split}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


class _DummyModel(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        _ = pretrained
        self.data_config = {"input_size": (3, 16, 16), "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_smoke_train_one_epoch_tiny_data(tmp_path: Path, monkeypatch) -> None:
    # disable wandb always in tests
    monkeypatch.setenv("WANDB_MODE", "disabled")

    processed = tmp_path / "processed"
    processed.mkdir()

    train_csv = _write_split(processed, "train", n=4)
    val_csv = _write_split(processed, "val", n=2)

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
            "data": {
                "train_dir": str(processed / "train"),
                "val_dir": str(processed / "val"),
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "train_limit": None,
                "val_limit": None,
                "num_workers": 0,
            },
            "train": {
                "device": "cpu",
                "seed": 123,
                "num_epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "step_size": 1,
                "gamma": 1.0,
                "pretrained": False,
                "wandb": False,
                "profile": False,
            },
            "model": {"model_name": "ignored", "num_classes": 2},
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
