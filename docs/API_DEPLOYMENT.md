# API Deployment Guide

Deploy your trained model as a REST API on Google Cloud Vertex AI Endpoints.

## Quick Start

```bash
# 1. Build and push training image (also used for API)
make push-train TAG=v1

# 2. Promote latest model to production
make promote-model VERSION=v1

# 3. Deploy API to Vertex AI Endpoints (reuses training image)
make deploy-api MODEL_VERSION=v1

# 4. Get endpoint URL and use the API
export ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west1 --filter="displayName:mlops-api-endpoint" --format="value(name)" --limit=1)
uv run inference path/to/images/ --api-url https://$ENDPOINT_ID-prediction.europe-west1-aiplatform.google.com
```

## Deployment

```bash
# Deploy with specific model version
make deploy-api MODEL_VERSION=v1

# Deploy with custom endpoint name
make deploy-api MODEL_VERSION=v1 ENDPOINT_NAME=my-api-endpoint
```

**What it does:**
- Reuses the training Docker image (same environment, better MLOps practice)
- Overrides entrypoint to run API instead of training script
- Deploys to Vertex AI Endpoints with GPU (T4 GPU, n1-standard-4 machine)
- Uses GCS mounting (`/gcs/` paths) - no model download needed!

**Note:** The API uses the same image as training - no separate build needed. Just ensure `make push-train` has been run first.

## Using the API

### CLI Tool

```bash
# Set endpoint ID
export ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west1 --filter="displayName:mlops-api-endpoint" --format="value(name)" --limit=1)
export API_URL=https://$ENDPOINT_ID-prediction.europe-west1-aiplatform.google.com

# Run predictions on a folder
uv run inference path/to/images/

# Results saved to predictions.csv
```

### Manual Requests

Vertex AI Endpoints uses a different request format. Use the prediction API:

```bash
# Get endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west1 --filter="displayName:mlops-api-endpoint" --format="value(name)" --limit=1)

# Health check (if your API exposes /health)
curl https://$ENDPOINT_ID-prediction.europe-west1-aiplatform.google.com/health

# Predict single image (requires authentication)
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -F "file=@image.jpg" \
  https://$ENDPOINT_ID-prediction.europe-west1-aiplatform.google.com/predict
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.9234,
  "label": "ai",
  "model_version": "v1"
}
```

**Labels:** `0` = human, `1` = ai

## Running Inference Tests

### Test on Preprocessed Data

```bash
# Set endpoint ID
export ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west1 --filter="displayName:mlops-api-endpoint" --format="value(name)" --limit=1)
export API_URL=https://$ENDPOINT_ID-prediction.europe-west1-aiplatform.google.com

# Run inference on preprocessed test folder
uv run inference data/processed/test/

# Results saved to predictions.csv
```

**CSV Output:**
```csv
image,prediction,label,confidence,model_version
test_001.jpg,1,ai,0.9234,v1
test_002.jpg,0,human,0.8743,v1
...
```

### Compare with Ground Truth

If you have test labels, compare predictions:

```bash
# Run inference
uv run inference data/processed/test/ --output api_predictions.csv

# Compare with ground truth (if you have test.csv)
python << 'EOF'
import pandas as pd

preds = pd.read_csv('api_predictions.csv')
truth = pd.read_csv('data/processed/test.csv')

# Merge on image name
merged = preds.merge(truth, left_on='image', right_on='file_name', how='inner')
accuracy = (merged['prediction'] == merged['label']).mean()
print(f"Accuracy: {accuracy:.2%}")
EOF
```

## Local Testing

```bash
# Download model
gsutil cp gs://mlops-training-stdne/models/latest/model.pt ./model.pt

# Run API locally
export MODEL_PATH=./model.pt
uv run uvicorn mlops_project.api:app --reload

# Test on preprocessed data
export API_URL=http://localhost:8000
uv run inference data/processed/test/
```

## Update Model

```bash
# Promote latest model (auto-finds newest run)
make promote-model VERSION=v2

# Or promote specific run
make promote-model RUN=2026-01-24/14-30-00 VERSION=v2

# Redeploy with new version (no rebuild needed - same image)
make deploy-api MODEL_VERSION=v2
```

**Note:** If you changed code/configs, rebuild the image first:
```bash
make push-train TAG=v1  # Rebuild training image (also used by API)
make deploy-api MODEL_VERSION=v2
```

## Configuration

Default settings:
- `ENDPOINT_NAME=mlops-api-endpoint`
- `MODEL_VERSION=v1`
- Machine: n1-standard-4, GPU: NVIDIA T4 (1 GPU)
- Auto-scales 0-1 replicas
- Uses same Docker image as training (reuses `mlops-trainer:v1`)
- Model path: `/gcs/mlops-training-stdne/models/$(MODEL_VERSION)/model.pt` (GCS mounted)

**Benefits of Vertex AI Endpoints:**
- ✅ Native GCS mounting - no model download needed
- ✅ Same GPU (T4) as training - consistent environment
- ✅ ML-optimized infrastructure
- ✅ Better monitoring and metrics
- ✅ Same ecosystem as training

Override in Makefile or command line.

## Troubleshooting

### Check Endpoint Status

```bash
# List endpoints
gcloud ai endpoints list --region=europe-west1

# Describe endpoint
gcloud ai endpoints describe ENDPOINT_ID --region=europe-west1

# Check deployed models
gcloud ai endpoints describe ENDPOINT_ID --region=europe-west1 --format="value(deployedModels)"
```

### View Logs

```bash
# Get endpoint logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" --limit=50
```

### Delete Endpoint

```bash
# Delete endpoint (removes all deployed models)
gcloud ai endpoints delete ENDPOINT_ID --region=europe-west1
```

---

For complete setup, see `GCP_VERTEX_AI_SETUP.md`.
For training workflow, see `GCP_WORKFLOW.md`.