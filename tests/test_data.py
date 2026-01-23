from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from mlops_project.data import MyDataset, NormalizeTransform, ResizeNormalizeTransform
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _write_dummy_dataset(root: Path, n: int = 3) -> Path:
    rows = []
    for i in range(n):
        img_name = f"img{i}.jpg"
        Image.new("RGB", (10 + i, 12 + i), color=(255, 255, 255)).save(root / img_name)
        rows.append({"file_name": img_name, "label": i % 2})
    csv_path = root / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_my_dataset_reads_one_sample(tmp_path: Path) -> None:
    csv_path = _write_dummy_dataset(tmp_path, n=1)

    dataset = MyDataset(tmp_path, csv_path=csv_path)
    assert isinstance(dataset, Dataset)

    x, y = dataset[0]
    assert isinstance(x, Image.Image)
    assert isinstance(y, int)
    assert y in (0, 1)


def test_my_dataset_limit(tmp_path: Path) -> None:
    csv_path = _write_dummy_dataset(tmp_path, n=5)

    dataset = MyDataset(tmp_path, csv_path=csv_path, limit=2)
    assert len(dataset) == 2


def test_normalize_transform_white_image_keeps_shape() -> None:
    img = Image.new("RGB", (11, 13), color=(255, 255, 255))
    transform = NormalizeTransform(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    x = transform(img)
    assert isinstance(x, Tensor)
    assert x.shape == (3, 13, 11)  # (C, H, W)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert torch.allclose(x, torch.ones_like(x), atol=1e-5)


def test_resize_normalize_transform_resizes() -> None:
    img = Image.new("RGB", (11, 13), color=(255, 255, 255))
    transform = ResizeNormalizeTransform(image_size=8, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    x = transform(img)
    assert isinstance(x, Tensor)
    assert x.shape == (3, 8, 8)
    assert torch.allclose(x, torch.ones_like(x), atol=1e-5)


def test_my_dataset_applies_transform_and_target_transform(tmp_path: Path) -> None:
    csv_path = _write_dummy_dataset(tmp_path, n=1)

    transform = ResizeNormalizeTransform(image_size=8, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    target_transform = lambda y: int(y) + 1  # noqa: E731

    dataset = MyDataset(tmp_path, csv_path=csv_path, transform=transform, target_transform=target_transform)
    x, y = dataset[0]

    assert isinstance(x, Tensor)
    assert x.shape == (3, 8, 8)
    assert y in (1, 2)


def test_my_dataset_supports_id_column_and_missing_label(tmp_path: Path) -> None:
    # create one image
    img_name = "abc.jpg"
    Image.new("RGB", (8, 8)).save(tmp_path / img_name)

    # test.csv-like: uses `id` and has no `label`
    csv_path = tmp_path / "test.csv"
    pd.DataFrame([{"id": img_name}]).to_csv(csv_path, index=False)

    ds = MyDataset(tmp_path, csv_path=csv_path)
    x, y = ds[0]
    assert isinstance(x, Image.Image)
    assert y == -1
