from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from mlops_project.data import MyDataset, TimmImageTransform
from mlops_project.model import Model

load_dotenv()


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


def _maybe_start_profiler(cfg: DictConfig, run_dir: Path) -> tuple[Any | None, Path | None]:
    do_profile = bool(cfg.train.get("profile", False))
    if not do_profile:
        return None, None

    prof_dir = run_dir / "profiler"
    prof_dir.mkdir(parents=True, exist_ok=True)

    activities: list[torch.profiler.ProfilerActivity] = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_dir)),
        record_shapes=True,
        with_stack=False,
    )
    prof.__enter__()
    return prof, prof_dir


def _maybe_stop_profiler(prof: torch.profiler.profile | None) -> None:
    if prof is not None:
        prof.__exit__(None, None, None)


def train_model(cfg: DictConfig) -> None:
    """Train a 2-class timm classifier on the dataset in `data/raw`."""
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    data_dir = Path(cfg.data.data_dir)
    device = _infer_device(cfg.train.device)
    _seed_everything(cfg.train.seed)

    # W&B init (disabled in CI via WANDB_MODE=disabled)
    run = None

    use_wandb = bool(cfg.train.get("wandb", True)) and os.getenv("WANDB_MODE", "").lower() != "disabled"
    if use_wandb:
        config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "mlops_project"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_dir.name,
            config=config,
        )
    # Optional torch profiling (enable with PROFILE=1)
    prof, prof_dir = _maybe_start_profiler(cfg, run_dir)

    try:
        model = Model(
            model_name=cfg.model.model_name,
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
        ).to(device)

        data_config = model.data_config
        input_size = data_config["input_size"]
        transform = TimmImageTransform(
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

                if prof is not None:
                    prof.step()

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

                    if prof is not None:
                        prof.step()

            train_loss /= max(1, train_total)
            train_acc = train_correct / max(1, train_total)
            val_loss /= max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            print(
                f"Epoch {epoch + 1}/{cfg.train.num_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

            if run is not None:
                run.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch + 1,
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

                if run is not None:
                    run.summary["best/val_acc"] = best_val_acc
                    run.summary["best/epoch"] = epoch + 1
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(str(best_path))
                    run.log_artifact(artifact)

        with metrics_path.open("w", newline="") as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
            )
            writer.writeheader()
            writer.writerows(metrics_history)

    finally:
        _maybe_stop_profiler(prof)

        if run is not None:
            # Upload profiler traces if enabled
            if prof_dir is not None and prof_dir.exists():
                artifact = wandb.Artifact("torch_profiler", type="profile")
                artifact.add_dir(str(prof_dir))
                run.log_artifact(artifact)

            run.finish()


@hydra.main(config_path=str(Path(__file__).parent.parent.parent / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for training."""
    train_model(cfg)


if __name__ == "__main__":
    main()
