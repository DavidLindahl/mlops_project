from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from mlops_project.data import MyDataset, NormalizeTransform, ResizeNormalizeTransform
from mlops_project.model import Model


def _infer_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_data_dir(cfg: DictConfig, project_root: Path) -> Path:
    data_dir = Path(cfg.data.data_dir)
    if data_dir.is_absolute():
        return data_dir
    return project_root / data_dir


def _latest_run_dir(base_dir: Path) -> Path:
    candidates: list[Path] = []
    if not base_dir.exists():
        raise FileNotFoundError(f"No runs directory found at {base_dir}")

    for run_dir in base_dir.glob("*/*"):
        if (run_dir / ".hydra" / "config.yaml").exists():
            candidates.append(run_dir)

    if not candidates:
        raise FileNotFoundError(f"No Hydra run directories found under {base_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_run_config(run_dir: Path) -> DictConfig:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config at {config_path}")
    return OmegaConf.load(config_path)


def _checkpoint_path(run_dir: Path) -> Path:
    preferred = run_dir / "checkpoints" / "best_model.pt"
    fallback = run_dir / "best_model.pt"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No best checkpoint found in {run_dir}")


def evaluate_run(run_dir: Path, device_override: str | None = None) -> Path:
    """Evaluate the best checkpoint from a run directory.

    Args:
        run_dir: Run directory containing a `.hydra` config and checkpoints.
        device_override: Optional device override ("cpu" or "cuda").

    Returns:
        Path to the saved metrics CSV file.
    """
    cfg = _load_run_config(run_dir)
    project_root = _project_root()
    raw_data_dir = _resolve_data_dir(cfg, project_root)
    processed_data_dir = Path(cfg.data.processed_dir)
    if not processed_data_dir.is_absolute():
        processed_data_dir = (project_root / processed_data_dir).resolve()
    use_processed = cfg.data.use_processed and processed_data_dir.exists()
    data_dir = processed_data_dir if use_processed else raw_data_dir
    device = _infer_device(device_override or cfg.train.device)

    model = Model(pretrained=False).to(device)
    checkpoint = torch.load(_checkpoint_path(run_dir), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    data_config = model.data_config
    input_size = data_config["input_size"]
    if use_processed:
        transform = NormalizeTransform(
            mean=list(data_config["mean"]),
            std=list(data_config["std"]),
        )
    else:
        transform = ResizeNormalizeTransform(
            image_size=int(input_size[-1]),
            mean=list(data_config["mean"]),
            std=list(data_config["std"]),
        )
    target_transform = lambda y: int(y)  # noqa: E731

    dataset = MyDataset(
        data_dir,
        limit=cfg.data.limit,
        transform=transform,
        target_transform=target_transform,
    )
    val_size = max(1, int(len(dataset) * cfg.train.val_fraction))
    train_size = len(dataset) - val_size
    _, val_ds = random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed),
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

    metrics_path = run_dir / "eval_metrics.csv"
    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["val_loss", "val_acc"])
        writer.writeheader()
        writer.writerow({"val_loss": val_loss, "val_acc": val_acc})

    print(f"Eval loss {val_loss:.4f} acc {val_acc:.4f}")
    print(f"Saved eval metrics to {metrics_path}")
    return metrics_path


def main() -> None:
    """Entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Optional folder name under 'models' containing a copied run directory.",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help="Explicit run directory to evaluate (overrides model_name).",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="Override device selection (e.g., cpu or cuda).",
    )
    args = parser.parse_args()

    project_root = _project_root()
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.model_name is not None:
        run_dir = project_root / "models" / args.model_name
    else:
        run_dir = _latest_run_dir(project_root / "reports" / "runs")

    evaluate_run(run_dir=run_dir, device_override=args.device)


if __name__ == "__main__":
    main()
