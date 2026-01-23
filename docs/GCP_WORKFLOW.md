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
# List available training runs
make list-runs

# Promote a specific run (auto-sets as latest)
make promote-model RUN=2026-01-23/10-47-37 VERSION=v1

# Or let it auto-generate version from date
make promote-model RUN=2026-01-23/10-47-37
```

**What it does**: Copies `best_model.pt` from the run to:
- `gs://mlops-training-stdne/models/<VERSION>/model.pt` (versioned)
- `gs://mlops-training-stdne/models/latest/model.pt` (for API)

### 5. Use Model in API

```python
# In your API code
MODEL_PATH = "/gcs/mlops-training-stdne/models/latest/model.pt"
# Or use versioned:
MODEL_PATH = "/gcs/mlops-training-stdne/models/v1/model.pt"
```

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

### Rebuild After Code Changes

```bash
# After changing code/configs, rebuild and push
make push-train TAG=v1

# Or use a new tag to keep old version
make push-train TAG=v2

# Update configs/vertex_train_config.yaml to use new tag if needed
```

## Configuration

Default values (can override in Makefile or as env vars):

- `PROJECT_ID=mlops-485010`
- `REGION=europe-west1`
- `REPO_NAME=mlops-training`
- `BUCKET=gs://mlops-training-stdne`
- `TAG=v1`

Override on command line:
```bash
make push-train TAG=v2 PROJECT_ID=my-project
```

## Troubleshooting

**Job fails immediately**: Check Docker image exists and config paths are correct.

**GPU quota exceeded**: Request quota increase or use CPU-only training.

**Model not found when promoting**: Verify run path exists with `make list-runs`.

**Build fails on Apple Silicon**: Makefile already uses `--platform linux/amd64`, should work.

## Full Workflow Example

```bash
# 1. Build images (first time or after code changes)
make push-train TAG=v1
make push-preprocess TAG=v1

# 2. Preprocess data
make submit-preprocess
# Wait for completion, check logs

# 3. Train model
make submit-train
# Monitor in W&B or logs

# 4. Promote best model
make list-runs
make promote-model RUN=2026-01-23/14-30-15 VERSION=v1

# 5. Use in API
# MODEL_PATH = "/gcs/mlops-training-stdne/models/latest/model.pt"
```

---

For initial setup, see `GCP_VERTEX_AI_SETUP.md`.
