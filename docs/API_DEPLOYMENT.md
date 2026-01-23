# Local API Usage

Run the model inference API locally.

## Quick Start

```bash
# 1. Download model
gsutil cp gs://mlops-training-stdne/models/latest/model.pt ./model.pt

# 2. Start API
export MODEL_PATH=./model.pt
uv run uvicorn mlops_project.api:app --reload --port 8000

# 3. Test with CLI
export API_URL=http://localhost:8000
uv run inference data/processed/test/
```

## Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Predict image
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

**Response:** `{"prediction": 1, "confidence": 0.9234, "label": "ai", "model_version": "latest"}`

**Labels:** `0` = human, `1` = ai
