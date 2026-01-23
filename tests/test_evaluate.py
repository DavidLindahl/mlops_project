import os
from pathlib import Path

import pytest
from mlops_project import evaluate as eval_mod
from omegaconf import DictConfig, OmegaConf


def test_infer_device_cpu() -> None:
    d = eval_mod._infer_device("cpu")
    assert d.type == "cpu"


def test_resolve_data_dir_relative(tmp_path: Path) -> None:
    cfg: DictConfig = OmegaConf.create({"data": {"data_dir": "data"}})
    out = eval_mod._resolve_data_dir(cfg, project_root=tmp_path)
    assert out == tmp_path / "data"


def test_resolve_data_dir_absolute(tmp_path: Path) -> None:
    abs_dir = tmp_path / "abs_data"
    cfg: DictConfig = OmegaConf.create({"data": {"data_dir": str(abs_dir)}})
    out = eval_mod._resolve_data_dir(cfg, project_root=Path("/does/not/matter"))
    assert out == abs_dir


def _make_run_dir(base: Path, name: str, mtime: int) -> Path:
    # evaluate._latest_run_dir expects base/*/* structure
    run_dir = base / name / "run"
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (run_dir / ".hydra" / "config.yaml").write_text("data:\n  data_dir: data\n", encoding="utf-8")
    os.utime(run_dir, (mtime, mtime))
    return run_dir


def test_latest_run_dir_picks_newest(tmp_path: Path) -> None:
    base = tmp_path / "runs"
    base.mkdir()

    older = _make_run_dir(base, "older", mtime=1000)
    newer = _make_run_dir(base, "newer", mtime=2000)

    chosen = eval_mod._latest_run_dir(base)
    assert chosen == newer
    assert chosen != older


def test_load_run_config_returns_dictconfig(tmp_path: Path) -> None:
    run_dir = tmp_path / "x" / "y"
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (run_dir / ".hydra" / "config.yaml").write_text("data:\n  data_dir: data\n", encoding="utf-8")

    cfg = eval_mod._load_run_config(run_dir)
    assert isinstance(cfg, DictConfig)


def test_load_run_config_missing_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "x" / "y"
    run_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        eval_mod._load_run_config(run_dir)


def test_checkpoint_path_prefers_checkpoints_dir(tmp_path: Path) -> None:
    run_dir = tmp_path
    (run_dir / "checkpoints").mkdir()
    preferred = run_dir / "checkpoints" / "best_model.pt"
    preferred.write_text("x", encoding="utf-8")

    assert eval_mod._checkpoint_path(run_dir) == preferred


def test_checkpoint_path_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path
    fallback = run_dir / "best_model.pt"
    fallback.write_text("x", encoding="utf-8")

    assert eval_mod._checkpoint_path(run_dir) == fallback


def test_checkpoint_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        eval_mod._checkpoint_path(tmp_path)
