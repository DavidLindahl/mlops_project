from __future__ import annotations

import csv
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlops_project.data import MyDataset, NormalizeTransform
from mlops_project.model import Model


def _infer_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path=str(Path(__file__).parent.parent.parent / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate a model copied into the models folder."""
    model_dir = Path(cfg.eval.output_dir) / cfg.eval.model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at {model_dir}")

    checkpoint_path = model_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {checkpoint_path}")

    data_dir = Path(cfg.data.val_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Validation data directory not found at {data_dir}")
    csv_path = cfg.data.val_csv
    data_limit = cfg.data.val_limit

    device = _infer_device(cfg.train.device)
    model = Model(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    data_config = model.data_config
    transform = NormalizeTransform(
        mean=list(data_config["mean"]),
        std=list(data_config["std"]),
    )
    target_transform = lambda y: int(y)  # noqa: E731

    val_ds = MyDataset(
        data_dir,
        csv_path=csv_path,
        limit=data_limit,
        transform=transform,
        target_transform=target_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Eval")
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

    val_loss /= max(1, val_total)
    val_acc = val_correct / max(1, val_total)

    metrics_path = model_dir / "eval_metrics.csv"
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["val_loss", "val_acc"])
        writer.writeheader()
        writer.writerow({"val_loss": val_loss, "val_acc": val_acc})

    print(f"Eval loss {val_loss:.4f} acc {val_acc:.4f}")
    print(f"Saved eval metrics to {metrics_path}")


if __name__ == "__main__":
    main()
