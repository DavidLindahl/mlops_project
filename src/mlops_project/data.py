from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class NormalizeTransform:
    """Normalize a tensor image using ImageNet statistics."""

    def __init__(self, mean: Iterable[float], std: Iterable[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        x = byte_tensor.view(image.height, image.width, 3).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        return (x - self.mean) / self.std


class ResizeNormalizeTransform:
    """Resize and normalize a PIL image for model input."""

    def __init__(self, image_size: int, mean: Iterable[float], std: Iterable[float]) -> None:
        self.image_size = image_size
        self.normalizer = NormalizeTransform(mean=mean, std=std)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        return self.normalizer(image)


class MyDataset(Dataset):
    """My custom dataset for AI vs Human generated images."""

    def __init__(
        self,
        data_path: str | Path,
        csv_path: str | Path,
        limit: int | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
        target_transform: Callable[[object], int] | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to the directory containing image files.
            csv_path: Path to the CSV file with annotations.
            limit: Optionally limit number of rows loaded from CSV for quick experiments.
            transform: Optional transform applied to the image (e.g., resize/normalize to tensors).
            target_transform: Optional transform applied to the label.
        """
        self.data_path = Path(data_path)
        self.annotations = pd.read_csv(csv_path, nrows=limit)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, int]:
        row = self.annotations.iloc[index]
        # Handle both 'file_name' (train/val) and 'id' (test) columns
        img_name = row.get("file_name", row.get("id"))
        image_path = self.data_path / img_name

        image: Tensor | Image.Image = Image.open(image_path).convert("RGB")
        label = int(row.get("label", -1))

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
