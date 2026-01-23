# GCP Vertex AI Workflow Guide

Quick reference for common GCP Vertex AI operations. Assumes setup is complete (see `GCP_VERTEX_AI_SETUP.md` for initial setup).

## Quick Commands

```bash
# See all available commands
make help
```

## Workflow: Preprocessing → Training → Model Promotion

### 1. Build and Push Images

```bash
# Build and push training image
make push-train TAG=v1

# Build and push preprocessing image (optional, lighter)
make push-preprocess TAG=v1
```

**Note**: If you change code/configs, rebuild and push before submitting jobs.

### 2. Preprocess Data

```bash
# Submit preprocessing job
make submit-preprocess

# Monitor logs (get JOB_ID from output)
gcloud ai custom-jobs stream-logs <JOB_ID> --region=europe-west1

# Verify processed data exists
gsutil ls gs://mlops-training-stdne/processed/
```

**What it does**: Processes raw data from `gs://mlops-training-stdne/raw/` and saves to `gs://mlops-training-stdne/processed/`.

### 3. Train Model

```bash
# Submit training job
make submit-train

# Monitor logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=europe-west1

# Check W&B dashboard (if enabled)
# https://wandb.ai/nikolajhj-technical-universty-of-denmark/mlops_project
```

**What it does**: Trains model on processed data, saves checkpoints and metrics to `gs://mlops-training-stdne/runs/<date>/<time>/`.

### 4. Promote Model to Production

```bash
# Promote latest model automatically (recommended)
make promote-model VERSION=v1

# Or promote a specific run
make promote-model RUN=2026-01-23/10-47-37 VERSION=v1

# List available runs to see what's available
make list-runs DATE=2026-01-23
```

**What it does**: Copies `best_model.pt` from the run to:
- `gs://mlops-training-stdne/models/<VERSION>/model.pt` (versioned)
- `gs://mlops-training-stdne/models/latest/model.pt` (for local API)

### 5. Run API Locally

```bash
# Download model
gsutil cp gs://mlops-training-stdne/models/latest/model.pt ./model.pt

# Run API locally
export MODEL_PATH=./model.pt
uv run uvicorn mlops_project.api:app --reload --port 8000

# Test with CLI tool
export API_URL=http://localhost:8000
uv run inference data/processed/test/
```

See [Local API Guide](API_DEPLOYMENT.md) for complete documentation.

## Common Operations

### Check Job Status

```bash
# List all jobs
gcloud ai custom-jobs list --region=europe-west1

# Get job details
gcloud ai custom-jobs describe <JOB_ID> --region=europe-west1

# Stream logs
gcloud ai custom-jobs stream-logs <JOB_ID> --region=europe-west1

# Cancel a job
gcloud ai custom-jobs cancel <JOB_ID> --region=europe-west1
```

### View Data in Bucket

```bash
# List runs
gsutil ls gs://mlops-training-stdne/runs/

# List models
gsutil ls gs://mlops-training-stdne/models/

# Download a model locally
gsutil cp gs://mlops-training-stdne/models/latest/model.pt ./model.pt
```