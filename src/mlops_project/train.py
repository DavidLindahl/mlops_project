from __future__ import annotations

import csv
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from mlops_project.data import MyDataset
from mlops_project.model import Model


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(cfg: DictConfig) -> None:
    """Train a 2-class timm classifier on the dataset in `data/raw`."""
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    data_dir = Path(cfg.data.data_dir)
    device = _infer_device(cfg.train.device)
    _seed_everything(cfg.train.seed)

    model = Model(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    ).to(device)
    data_config = model.data_config
    input_size = data_config["input_size"]
    transform = ViTImageTransform(
        image_size=int(input_size[-1]),
        mean=list(data_config["mean"]),
        std=list(data_config["std"]),
    )
    target_transform = lambda y: int(y)  # noqa: E731

    dataset = MyDataset(data_dir, transform=transform, target_transform=target_transform)
    val_size = max(1, int(len(dataset) * cfg.train.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.train.step_size, gamma=cfg.train.gamma)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"
    metrics_history: list[dict[str, float | int]] = []

    for epoch in range(cfg.train.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.num_epochs} [train]")
        for images, labels in train_bar:
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
            train_bar.set_postfix(loss=loss.item())

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.train.num_epochs} [val]")
            for images, labels in val_bar:
                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                logits = model(images)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)
                val_bar.set_postfix(loss=loss.item())

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch + 1}/{cfg.train.num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        metrics_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        last_checkpoint = {
            "model_name": cfg.model.model_name,
            "num_classes": cfg.model.num_classes,
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "val_acc": val_acc,
        }
        torch.save(last_checkpoint, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint = {
                "model_name": cfg.model.model_name,
                "num_classes": cfg.model.num_classes,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }
            torch.save(best_checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path} (val acc {val_acc:.4f})")

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(metrics_history)


@hydra.main(config_path=str(Path(__file__).parent.parent.parent / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for training."""
    train_model(cfg)


if __name__ == "__main__":
    main()
