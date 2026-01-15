from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from mlops_project.data import MyDataset
from mlops_project.model import Model


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for model training."""

    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("models")
    model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    batch_size: int = 16
    num_epochs: int = 3
    num_workers: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.1
    seed: int = 42
    device: str | None = None


class ViTImageTransform:
    """Minimal PIL->Tensor transform without requiring torchvision."""

    def __init__(self, image_size: int, mean: list[float], std: list[float]) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        x = byte_tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1).contiguous()
        x = x.to(dtype=torch.float32).div_(255.0)
        return (x - self.mean) / self.std


def _infer_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(config: TrainConfig | None = None) -> None:
    """Train a 2-class timm classifier on the dataset in `data/raw`."""
    config = config or TrainConfig()
    device = _infer_device(config.device)
    _seed_everything(config.seed)

    model = Model(model_name=config.model_name, num_classes=2, pretrained=True).to(device)
    data_config = model.data_config
    input_size = data_config["input_size"]
    transform = ViTImageTransform(
        image_size=int(input_size[-1]),
        mean=list(data_config["mean"]),
        std=list(data_config["std"]),
    )
    target_transform = lambda y: int(y)  # noqa: E731

    dataset = MyDataset(config.data_dir, transform=transform, target_transform=target_transform)
    val_size = max(1, int(len(dataset) * config.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    LR = config.lr
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    safe_name = config.model_name.replace("/", "_").replace(".", "_")
    best_path = config.output_dir / f"{safe_name}_best.pt"

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                logits = model(images)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": config.model_name,
                    "num_classes": 2,
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "config": config,
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path} (val acc {val_acc:.4f})")

if __name__ == "__main__":
    train()
