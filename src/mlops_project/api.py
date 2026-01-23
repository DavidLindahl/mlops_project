"""FastAPI inference service for image classification."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from mlops_project.data import NormalizeTransform
from mlops_project.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Image Classifier API",
    description="Binary classification: AI-generated vs Human-generated images",
    version="1.0.0",
)

# Global model state
model: Model | None = None
transform: NormalizeTransform | None = None
device: torch.device = torch.device("cpu")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: int
    confidence: float
    label: str
    model_version: str


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str
    model_loaded: bool
    model_version: str
    device: str


def load_model_from_gcs(model_path: str) -> tuple[Model, NormalizeTransform]:
    """Load model from GCS path (Vertex AI mounts GCS at /gcs/).

    Args:
        model_path: Path to model checkpoint (/gcs/bucket/path for Vertex AI, or local path)

    Returns:
        Tuple of (model, transform)
    """
    logger.info(f"Loading model from: {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = Model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data_config = model.data_config
    transform = NormalizeTransform(
        mean=list(data_config["mean"]),
        std=list(data_config["std"]),
    )

    logger.info(f"Model loaded successfully. Input size: {data_config['input_size']}")
    logger.info(f"Checkpoint info - Epoch: {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 'N/A')}")

    return model, transform


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    global model, transform, device

    model_path = os.getenv("MODEL_PATH", "/gcs/mlops-training-stdne/models/latest/model.pt")
    model_version = os.getenv("MODEL_VERSION", "latest")

    logger.info(f"Starting API with model version: {model_version}")
    logger.info(f"Model path: {model_path}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    if use_cuda:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        model, transform = load_model_from_gcs(model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "message": "MLOps Image Classifier API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image file)",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=os.getenv("MODEL_VERSION", "latest"),
        device=str(device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Predict if image is AI-generated or human-generated.

    Args:
        file: Image file to classify

    Returns:
        Prediction with confidence score

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG/PNG supported.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        input_size = model.data_config["input_size"]
        image = image.resize((input_size[1], input_size[2]))

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        label_map = {0: "human", 1: "ai"}
        label = label_map.get(pred_class, "unknown")

        logger.info(f"Prediction: {label} (confidence: {confidence:.2%})")

        return PredictionResponse(
            prediction=pred_class,
            confidence=confidence,
            label=label,
            model_version=os.getenv("MODEL_VERSION", "latest"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
