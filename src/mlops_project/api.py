
from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from torch.nn import functional as F

from mlops_project.data import TimmImageTransform
from mlops_project.model import Model

MODEL_PATH = Path("models/model.pth")


class PredictResponse(BaseModel):
    """Response schema for predictions."""

    class_id: int
    probabilities: list[float]


def _infer_device() -> torch.device:
    """Infer the device for inference."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[Model, TimmImageTransform]:
    """Load the model and preprocessing transform from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model onto.

    Returns:
        A tuple with the model and its preprocessing transform.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If the checkpoint format is unexpected.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint should be a dict containing model metadata and state_dict.")

    model_name = checkpoint.get("model_name", "tf_efficientnetv2_s.in21k_ft_in1k")
    num_classes = int(checkpoint.get("num_classes", 2))
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing a state_dict.")

    model = Model(model_name=model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_config = model.data_config
    input_size = data_config["input_size"]
    transform = TimmImageTransform(
        image_size=int(input_size[-1]),
        mean=list(data_config["mean"]),
        std=list(data_config["std"]),
    )
    return model, transform


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    device = _infer_device()
    model, transform = _load_model(MODEL_PATH, device=device)
    app.state.device = device
    app.state.model = model
    app.state.transform = transform
    yield
    del app.state.model
    del app.state.transform
    del app.state.device


app = FastAPI(title="MLOps Project API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Predict the class for an uploaded image.

    Args:
        file: Uploaded image file.

    Returns:
        Prediction response with class id and probabilities.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        payload = await file.read()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    transform: TimmImageTransform = app.state.transform
    device: torch.device = app.state.device
    model: Model = app.state.model

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    class_id = int(torch.argmax(probs).item())
    return PredictResponse(class_id=class_id, probabilities=[float(p) for p in probs])
