from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class TimmImageTransform:
    """Minimal PIL-to-tensor transform without requiring torchvision."""

    def __init__(self, image_size: int, mean: list[float], std: list[float]) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> Tensor:
        image = image.resize((self.image_size, self.image_size))
        byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        x = byte_tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        return (x - self.mean) / self.std


class MyDataset(Dataset):
    """My custom dataset for AI vs Human generated images."""

    def __init__(
        self,
        data_path: str | Path,
        limit: int | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
        target_transform: Callable[[object], int] | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.annotations = pd.read_csv(self.data_path / "train.csv", nrows=limit)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, int]:
        img_name = self.annotations.iloc[index]["file_name"]
        image_path = self.data_path / img_name

        pil_image = Image.open(image_path).convert("RGB")  # PIL only
        label = int(self.annotations.iloc[index]["label"])

        image: Tensor | Image.Image
        if self.transform is not None:
            image = self.transform(pil_image)  # OK: transform expects PIL
        else:
            image = pil_image

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
