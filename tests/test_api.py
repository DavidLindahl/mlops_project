from __future__ import annotations

import io

import mlops_project.api as api_mod
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn


class _DummyTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        # return (C,H,W) float32 in [0,1]
        x = torch.zeros((3, image.height, image.width), dtype=torch.float32)
        return x


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.data_config = {"input_size": (3, 8, 8), "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # always predict class 1 with high confidence
        logits = torch.tensor([[0.0, 10.0]], dtype=torch.float32).repeat(x.shape[0], 1)
        return logits


@pytest.fixture(autouse=True)
def _reset_api_globals() -> None:
    api_mod.model = None
    api_mod.transform = None
    api_mod.device = torch.device("cpu")


def test_root_ok() -> None:
    client = TestClient(api_mod.app)
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "endpoints" in body


def test_health_unhealthy_when_model_not_loaded() -> None:
    client = TestClient(api_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False
    assert body["status"] == "unhealthy"


def test_predict_returns_503_if_model_not_loaded() -> None:
    client = TestClient(api_mod.app)

    img = Image.new("RGB", (8, 8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 503


def test_predict_rejects_invalid_content_type() -> None:
    api_mod.model = _DummyModel()
    api_mod.transform = _DummyTransform()

    client = TestClient(api_mod.app)
    r = client.post("/predict", files={"file": ("x.txt", b"not-an-image", "text/plain")})
    assert r.status_code == 400


def test_predict_success() -> None:
    api_mod.model = _DummyModel()
    api_mod.transform = _DummyTransform()

    client = TestClient(api_mod.app)

    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("human", "ai", "unknown")


def test_load_model_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        api_mod.load_model("does-not-exist.pt")
