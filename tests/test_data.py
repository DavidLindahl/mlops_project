from pathlib import Path

import pandas as pd
import torch
from mlops_project.data import MyDataset, TimmImageTransform
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def _write_dummy_dataset(root: Path, n: int = 3) -> None:
    rows = []
    for i in range(n):
        img_name = f"img{i}.jpg"
        Image.new("RGB", (10 + i, 12 + i)).save(root / img_name)
        rows.append({"file_name": img_name, "label": i % 2})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


def test_my_dataset_reads_one_sample(tmp_path: Path) -> None:
    _write_dummy_dataset(tmp_path, n=1)

    dataset = MyDataset(tmp_path)
    assert isinstance(dataset, Dataset)
    x, y = dataset[0]
    assert isinstance(x, Image.Image)
    assert isinstance(y, int)
    assert y in (0, 1)


def test_my_dataset_limit(tmp_path: Path) -> None:
    _write_dummy_dataset(tmp_path, n=5)

    dataset = MyDataset(tmp_path, limit=2)
    assert len(dataset) == 2


def test_timm_image_transform_shape_dtype() -> None:
    img = Image.new("RGB", (11, 13), color=(255, 255, 255))
    transform = TimmImageTransform(image_size=8, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    x = transform(img)
    assert isinstance(x, Tensor)
    assert x.shape == (3, 8, 8)
    assert x.dtype == torch.float32
    # white image -> values close to 1.0 after /255
    assert torch.isfinite(x).all()
    assert torch.allclose(x, torch.ones_like(x), atol=1e-5)


def test_my_dataset_applies_transform_and_target_transform(tmp_path: Path) -> None:
    _write_dummy_dataset(tmp_path, n=1)

    transform = TimmImageTransform(image_size=8, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    target_transform = lambda y: int(y) + 1  # noqa: E731

    dataset = MyDataset(tmp_path, transform=transform, target_transform=target_transform)
    x, y = dataset[0]

    assert isinstance(x, Tensor)
    assert x.shape == (3, 8, 8)
    assert y in (1, 2)
