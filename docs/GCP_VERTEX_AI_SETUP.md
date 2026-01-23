# GCP Vertex AI Training Setup Guide

This guide walks you through transitioning your existing Docker workflow to run on **Google Cloud Vertex AI Training**. It follows the DTU MLOps approach where GCP handles provisioning, execution, and shutdown of hardware automatically.

---

> **Important**: This project uses DVC with `gs://dtu-mlops-first-bucket` as a **read-only** data source. 
> You will create your own GCS bucket for storing processed data and training outputs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Environment & Project Setup](#phase-1-environment--project-setup)
3. [Phase 2: Data Strategy with Cloud Storage](#phase-2-data-strategy-with-cloud-storage)
4. [Phase 3: Build the Vertex AI Container](#phase-3-build-the-vertex-ai-container)
5. [Phase 4: Create the Vertex AI Config](#phase-4-create-the-vertex-ai-config)
6. [Phase 5: Submit Training Jobs](#phase-5-submit-training-jobs)
7. [Phase 6: Evaluation and Inference](#phase-6-evaluation-and-inference)
8. [Phase 7: Monitoring and Model Artifacts](#phase-7-monitoring-and-model-artifacts)
9. [Phase 8: Troubleshooting](#phase-8-troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- Docker installed locally
- A GCP account with billing enabled
- Read access to `gs://dtu-mlops-first-bucket` (public, used for initial data pull via DVC)

---

## Phase 1: Environment & Project Setup

### 1.1 Authenticate and Configure GCloud

```bash
# Login to your GCP account
gcloud auth login

# Application default credentials (needed for DVC and gsutil)
gcloud auth application-default login

# Set your project (replace with your actual project ID)
gcloud config set project mlops-485010

# Verify configuration
gcloud config list
```

### 1.2 Enable Required APIs

```bash
gcloud services enable aiplatform.googleapis.com \
                       artifactregistry.googleapis.com \
                       compute.googleapis.com \
                       storage.googleapis.com
```

### 1.3 Create Your Own Cloud Storage Bucket

Since `gs://dtu-mlops-first-bucket` is read-only (used as the DVC source), you need your own bucket for:
- Storing processed data
- Saving training outputs and model checkpoints
- Optionally mirroring raw data for faster access

```bash
# Create your bucket (name must be globally unique)
# Replace mlops-training-stdne with something unique, e.g., mlops-[YOUR_NAME]-data
gcloud storage buckets create gs://mlops-training-stdne \
    --location=europe-west1 \
    --uniform-bucket-level-access

# Verify bucket was created
gsutil ls gs://mlops-training-stdne/
```

**Recommended bucket naming**: `mlops-training-mlops-485010` or `[YOUR_NAME]-mlops-data`

### 1.4 Create an Artifact Registry Repository

This will store your Docker images:

```bash
gcloud artifacts repositories create mlops-training \
    --repository-format=docker \
    --location=europe-west1 \
    --description="Docker repository for ML training images"
```

### 1.5 Configure Docker Authentication

```bash
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

---

## Phase 2: Data Strategy with Cloud Storage

### Understanding the Current DVC Setup

The project uses DVC with `gs://dtu-mlops-first-bucket` as the remote. This bucket stores files using **content-addressable storage** (MD5 hashes):

```
gs://dtu-mlops-first-bucket/files/md5/XX/YYYYYYYY...
```

This means you **cannot** directly access files like `/gcs/dtu-mlops-first-bucket/raw/image.jpg`. You must use `dvc pull` to reconstruct the original file structure.

### 2.1 Configure DVC with Your Bucket

Add your bucket as a new DVC remote for pushing data:

```bash
# Add your bucket as a new remote called "trainremote"
uv run python -m dvc remote add trainremote gs://mlops-training-stdne

# (Optional) Set your remote as the default for pushing
uv run python -m dvc remote default trainremote

# Verify the configuration
cat .dvc/config
```

Your `.dvc/config` should now look like:

```ini
[core]
    remote = trainremote
    no_scm = true
    analytics = false

['remote "storage"']
    url = gs://dtu-mlops-first-bucket
    jobs = 1

['remote "trainremote"']
    url = gs://mlops-training-stdne
```

### 2.2 Ensure DVC Dependencies are Installed

Before pulling data, ensure `dvc-gs` is properly installed:

```bash
# Sync dependencies to ensure dvc[gs] extras are installed
uv sync

# Verify dvc-gs is available
uv run python -c "import dvc_gs; print('dvc-gs is installed')"
```

If you get an import error, explicitly install the GS extra:

```bash
# Explicitly install dvc-gs package
uv add "dvc-gs>=3.0.2"
uv sync
```

**Important**: Use `python -m dvc` instead of just `dvc` to ensure DVC uses the correct Python environment where `dvc-gs` is installed.

### 2.3 Pull Data from Source Bucket

Pull the raw data from the original (read-only) bucket:

```bash
# Pull data from the original DVC remote
# Use 'python -m dvc' to ensure correct Python environment
uv run python -m dvc pull -r storage
```

This downloads:
- `data/raw/train_data/` (~80K images, ~4.6GB)
- `data/raw/test_data/` (~5.5K images, ~7GB)

**Note**: This requires authentication to GCS. Make sure you've run `gcloud auth application-default login` (see Phase 1.1).

### 2.4 Push Data to Your Bucket

Now push the data to your own bucket so you have full control:

```bash
# Push data to your bucket
# Use 'python -m dvc' to ensure correct Python environment
uv run python -m dvc push -r trainremote
```

### 2.5 Upload CSV Files for Preprocessing

The image data is already in your bucket (from DVC push), but you need to upload the CSV index files:

```bash
# Upload only the CSV files
gsutil cp data/raw/train.csv gs://mlops-training-stdne/raw/
gsutil cp data/raw/test.csv gs://mlops-training-stdne/raw/

# Verify the CSV files are uploaded
gsutil ls gs://mlops-training-stdne/raw/*.csv
```

**Note**: You don't need to upload the image directories again if they're already in the bucket from `dvc push`. The preprocessing script needs:
1. The CSV files (uploaded above)
2. The image files referenced in the CSVs (already in bucket from DVC)

Your bucket structure should now be:

```
gs://mlops-training-stdne/
├── files/md5/...              # DVC cache (image files)
└── raw/                        # CSV index files
    ├── train.csv              # Points to image files
    └── test.csv               # Points to image files
```

### 2.6 Verify Your Data Setup

```bash
# List bucket contents
gsutil ls -r gs://mlops-training-stdne/

# Check raw data exists (for preprocessing)
gsutil ls gs://mlops-training-stdne/raw/train_data/ | head -5
gsutil cat gs://mlops-training-stdne/raw/train.csv | head -5

# After preprocessing, check processed data
gsutil ls gs://mlops-training-stdne/processed/train/ | head -5
gsutil cat gs://mlops-training-stdne/processed/train.csv | head -5
```

---

## Phase 3: Build the Vertex AI Container

### 3.0 Quick Start with Makefile

For convenience, a Makefile provides simple commands for common workflows:

```bash
# See all available commands
make help

# Build and push training image
make push-train TAG=v1

# Build and push preprocessing image
make push-preprocess TAG=v1

# Submit training job
make submit-train

# Submit preprocessing job
make submit-preprocess

# Promote a model to production
make promote-model RUN=2026-01-23/10-47-37 VERSION=v1

# List available runs
make list-runs
```

All commands support environment variables (set in Makefile or override):
- `PROJECT_ID` (default: `mlops-485010`)
- `REGION` (default: `europe-west1`)
- `REPO_NAME` (default: `mlops-training`)
- `BUCKET` (default: `gs://mlops-training-stdne`)
- `TAG` (default: `v1`)

**Note**: The manual steps below still work if you prefer more control.

### 3.1 Create the Vertex AI Dockerfile

Edit `dockerfiles/vertex_train.dockerfile`:

```dockerfile
# Base image with CUDA support for GPU training
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Create a virtual environment and install dependencies
# Note: We need GPU-compatible PyTorch, not CPU version
RUN uv venv && \
    uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 && \
    uv sync --frozen --no-install-project

# Copy source code and configs
COPY configs configs/
COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

# Install the project
RUN uv sync --frozen

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Vertex AI calls this entrypoint
ENTRYPOINT ["uv", "run", "train"]
```

### 3.2 Build and Push the Image

**Option A: Using Makefile (Recommended)**

```bash
# Build and push training image
make push-train TAG=v1

# Build only (without pushing)
make build-train TAG=v1
```

**Option B: Manual Docker Commands**

```bash
# Set your project ID
export PROJECT_ID=mlops-485010
export REGION=europe-west1
export REPO_NAME=mlops-training
export IMAGE_NAME=mlops-trainer
export IMAGE_TAG=v1

# Full image URI
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the training image (includes PyTorch with CUDA)
# Build for linux/amd64 (required for Vertex AI, even on Apple Silicon Macs)
docker build --platform linux/amd64 -f dockerfiles/vertex_train.dockerfile -t ${IMAGE_URI} .

# Push to Artifact Registry
docker push ${IMAGE_URI}
```

### 3.3 Build and Push the Preprocessing Image (Optional but Recommended)

For faster preprocessing builds and smaller image size:

**Option A: Using Makefile (Recommended)**

```bash
# Build and push preprocessing image
make push-preprocess TAG=v1
```

**Option B: Manual Docker Commands**

```bash
# Use same variables, different image name
export PREPROCESS_IMAGE_NAME=mlops-preprocessor
export PREPROCESS_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PREPROCESS_IMAGE_NAME}:${IMAGE_TAG}"

# Build the preprocessing image (lighter, no CUDA/PyTorch)
docker build --platform linux/amd64 \
    -f dockerfiles/vertex_preprocess.dockerfile \
    -t ${PREPROCESS_IMAGE_URI} .

# Push to Artifact Registry
docker push ${PREPROCESS_IMAGE_URI}
```

**Note**: If you skip this step, you can use the training image (`mlops-trainer:v1`) for preprocessing - it will work, just larger. Make sure to update `configs/vertex_preprocess_config.yaml` to use `mlops-preprocessor:v1` if you build it.

### 3.4 Verify the Images

```bash
# List all images in your repository
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}

# You should see both:
# - mlops-trainer:v1 (for training)
# - mlops-preprocessor:v1 (for preprocessing, if you built it)
```

---

## Phase 4: Create the Vertex AI Config

### 4.1 Create Training Config File

Create `configs/vertex_train_config.yaml`:

```yaml
# Vertex AI Custom Job Configuration
# Submit with: gcloud ai custom-jobs create --region=europe-west1 --display-name="mlops_training" --config=configs/vertex_train_config.yaml
#
# IMPORTANT: Replace mlops-training-stdne if using a different bucket name!
# IMPORTANT: Add your WANDB_API_KEY (get from: https://wandb.ai/authorize)

workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-8
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: europe-west1-docker.pkg.dev/mlops-485010/mlops-training/mlops-trainer:v1
    env:
      # W&B API key for experiment tracking (REPLACE WITH YOUR KEY)
      - name: WANDB_API_KEY
        value: "YOUR_WANDB_API_KEY_HERE"
      - name: WANDB_PROJECT
        value: "mlops-vertex-training"
      # Uncomment to disable W&B if you don't want to use it:
      # - name: WANDB_MODE
      #   value: "disabled"
    args:
      # Hydra overrides for GCS paths (use YOUR bucket with processed data)
      - "data.train_dir=/gcs/mlops-training-stdne/processed/train"
      - "data.val_dir=/gcs/mlops-training-stdne/processed/val"
      - "data.train_csv=/gcs/mlops-training-stdne/processed/train.csv"
      - "data.val_csv=/gcs/mlops-training-stdne/processed/val.csv"
      # Training parameters
      - "train.num_epochs=10"
      - "train.batch_size=32"
      # Output directory - saves checkpoints and metrics to YOUR bucket
      - "hydra.run.dir=/gcs/mlops-training-stdne/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
```

**W&B Setup:**
1. Get your API key from: https://wandb.ai/authorize
2. Replace `YOUR_WANDB_API_KEY_HERE` with your actual key
3. Or disable W&B by uncommenting the `WANDB_MODE: disabled` env var

### 4.2 Create Preprocessing Config

Create `configs/vertex_preprocess_config.yaml`:

```yaml
# Vertex AI Preprocessing Job Configuration
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
    # No GPU needed for preprocessing
  replicaCount: 1
  containerSpec:
    imageUri: europe-west1-docker.pkg.dev/mlops-485010/mlops-training/mlops-preprocessor:v1
    command:
      - "uv"
      - "run"
      - "preprocess"
    args:
      # Hydra overrides for GCS paths
      - "data.data_dir=/gcs/mlops-training-stdne/raw"
      - "data.processed_dir=/gcs/mlops-training-stdne/processed"
      - "data.image_size=224"
      # Optionally limit data for quick testing
      # - "data.train_limit=1000"
      # - "data.val_limit=200"
```

**Note**: The preprocessing Dockerfile (`dockerfiles/vertex_preprocess.dockerfile`) is already created and copies the `configs/` directory. The `args` above are Hydra overrides that override values in `configs/data.yaml`.

### 4.3 Config for Different GPU Types

**For T4 GPU (Cost-effective):**
```yaml
machineSpec:
  machineType: n1-standard-8
  acceleratorType: NVIDIA_TESLA_T4
  acceleratorCount: 1
```

**For V100 GPU (Faster training):**
```yaml
machineSpec:
  machineType: n1-standard-8
  acceleratorType: NVIDIA_TESLA_V100
  acceleratorCount: 1
```

**For A100 GPU (Highest performance):**
```yaml
machineSpec:
  machineType: a2-highgpu-1g
  acceleratorType: NVIDIA_TESLA_A100
  acceleratorCount: 1
```

**CPU-only (For testing/preprocessing):**
```yaml
machineSpec:
  machineType: n1-standard-4
```

---

## Phase 5: Submit Jobs to Vertex AI

### 5.1 Submit a Preprocessing Job

**Option A: Using Makefile (Recommended)**

```bash
# Submit preprocessing job (auto-generates name with timestamp)
make submit-preprocess
```

**Option B: Manual gcloud Command**

```bash
gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name="mlops_preprocessing_$(date +%Y%m%d_%H%M%S)" \
    --config=configs/vertex_preprocess_config.yaml
```

**Monitor and verify:**

```bash
# Stream logs
gcloud ai custom-jobs stream-logs [JOB_ID] --region=europe-west1

# Verify processed data after completion
gsutil ls gs://mlops-training-stdne/processed/
```

### 5.2 Submit a Training Job

**Option A: Using Makefile (Recommended)**

```bash
# Submit training job (auto-generates name with timestamp)
make submit-train
```

**Option B: Manual gcloud Command**

```bash
# Config file already has correct project ID (mlops-485010), then:
gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name="mlops_experiment_001" \
    --config=configs/vertex_train_config.yaml
```

### 5.3 Submit with Inline Overrides

You can also override config values directly:

```bash
gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name="mlops_experiment_002" \
    --config=configs/vertex_train_config.yaml \
    --args="train.num_epochs=20,train.lr=0.0001"
```

### 5.4 Quick Local Test Before Submitting

Test your container locally by simulating the `/gcs/` mount that Vertex AI provides:

```bash
# Create a test directory structure mimicking the GCS mount
mkdir -p /tmp/gcs/mlops-training-stdne/processed

# Copy your processed data (for testing)
cp -r data/processed/* /tmp/gcs/mlops-training-stdne/processed/

# Run container locally (replace mlops-training-stdne)
docker run --rm \
    -v /tmp/gcs:/gcs \
    ${IMAGE_URI} \
    data.train_dir=/gcs/mlops-training-stdne/processed/train \
    data.val_dir=/gcs/mlops-training-stdne/processed/val \
    data.train_csv=/gcs/mlops-training-stdne/processed/train.csv \
    data.val_csv=/gcs/mlops-training-stdne/processed/val.csv \
    data.train_limit=50 \
    data.val_limit=10 \
    train.num_epochs=1
```

---

## Phase 6: Evaluation and Inference

After training, evaluate your model and run inference on test data.

### 6.1 Run Evaluation on Vertex AI (RECOMMENDED)

Create `configs/vertex_eval_config.yaml`:

```yaml
# Vertex AI Evaluation Job Configuration
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: europe-west1-docker.pkg.dev/mlops-485010/mlops-training/mlops-trainer:v1
    command:
      - "uv"
      - "run"
      - "eval"
    args:
      # Path to validation/test data
      - "data.val_dir=/gcs/mlops-training-stdne/processed/val"
      - "data.val_csv=/gcs/mlops-training-stdne/processed/val.csv"
      # Specify which training run to evaluate (update this!)
      - "eval.model_name=2026-01-22/12-30-45"
      - "eval.output_dir=/gcs/mlops-training-stdne/runs"
```

**Submit evaluation job:**

```bash
# After training completes, note the run directory (e.g., 2026-01-22/12-30-45)
# Update the eval.model_name in the config above, then:

gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name="mlops_eval_$(date +%Y%m%d_%H%M%S)" \
    --config=configs/vertex_eval_config.yaml
```

**View evaluation results:**

```bash
# Check metrics
gsutil cat gs://mlops-training-stdne/runs/2026-01-22/12-30-45/eval_metrics.csv

# Download entire evaluation output
gsutil cp gs://mlops-training-stdne/runs/2026-01-22/12-30-45/eval_metrics.csv ./
```

### 6.2 Local Evaluation (Alternative)

If you prefer to evaluate locally:

```bash
# Download the model
gsutil -m cp -r gs://mlops-training-stdne/runs/2026-01-22/12-30-45/ models/my_model/

# Run evaluation
uv run eval \
    eval.model_name=my_model \
    eval.output_dir=models \
    data.val_dir=data/processed/val \
    data.val_csv=data/processed/val.csv
```

### 6.3 Simple Inference Example

For single-image inference, your existing `mlops_project` likely has model loading capabilities. Here's a minimal approach:

```python
# Quick inference script (run locally or add to your codebase)
import torch
from PIL import Image
from mlops_project.model import Model
from mlops_project.data import NormalizeTransform

# Load checkpoint
checkpoint = torch.load("best_model.pt", map_location="cpu")
model = Model(pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Setup transform
data_config = model.data_config
transform = NormalizeTransform(
    mean=list(data_config["mean"]),
    std=list(data_config["std"]),
)

# Run inference
image = Image.open("test_image.jpg").convert("RGB")
image = image.resize(data_config["input_size"][:2])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image_tensor)
    pred_class = logits.argmax(dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()

print(f"Predicted class: {pred_class}, Confidence: {confidence:.2%}")
```

### 6.4 Promote Model to Production

After training, promote your best model to a dedicated `models/` folder for API inference:

**Using Makefile (Recommended):**

```bash
# List available runs
make list-runs

# Promote a specific run to versioned folder (auto-sets as latest)
make promote-model RUN=2026-01-23/10-47-37 VERSION=v1

# If VERSION not specified, uses date from RUN path
make promote-model RUN=2026-01-23/10-47-37
```

**Manual Method:**

```bash
# Copy best model to versioned folder
gsutil cp \
  gs://mlops-training-stdne/runs/2026-01-23/10-47-37/checkpoints/best_model.pt \
  gs://mlops-training-stdne/models/v1/model.pt

# Set as latest for API
gsutil cp \
  gs://mlops-training-stdne/models/v1/model.pt \
  gs://mlops-training-stdne/models/latest/model.pt
```

**Recommended Folder Structure:**

```
gs://mlops-training-stdne/
├── runs/                    # Training runs (timestamped)
│   └── 2026-01-23/
│       └── 10-47-37/
│           └── checkpoints/
│               ├── best_model.pt
│               └── last_model.pt
└── models/                  # Production models
    ├── latest/              # Currently deployed (for API)
    │   └── model.pt
    └── v1/                  # Versioned models
        └── model.pt
```

**Use in API:**

```python
# In your API code, load from GCS
MODEL_PATH = "/gcs/mlops-training-stdne/models/latest/model.pt"
# Or versioned:
MODEL_PATH = "/gcs/mlops-training-stdne/models/v1/model.pt"
```

### 6.5 Best Practices

- ✅ **Run evaluation on Vertex AI** - Same environment as training, reproducible
- ✅ **Store all artifacts in GCS** - Checkpoints, metrics, predictions
- ✅ **Version your models** - Use timestamps or run IDs (e.g., `2026-01-22/12-30-45`)
- ✅ **Track metrics** - Use the CSV outputs from training and evaluation
- ✅ **Evaluate before deploying** - Always check performance on held-out data

---

## Phase 7: Monitoring and Model Artifacts

### 7.1 Monitor Job Progress

**Via CLI:**
```bash
# List running jobs
gcloud ai custom-jobs list --region=europe-west1

# Get job details
gcloud ai custom-jobs describe [JOB_ID] --region=europe-west1

# Stream logs
gcloud ai custom-jobs stream-logs [JOB_ID] --region=europe-west1
```

**Via Console:**
1. Go to [Vertex AI Console](https://console.cloud.google.com/vertex-ai/training/custom-jobs)
2. Select your region (europe-west1)
3. Click on your job to see real-time logs

### 7.2 Access Model Artifacts

Your trained models will be saved to the GCS path specified in hydra config (your bucket):

```bash
# List training runs
gsutil ls gs://mlops-training-stdne/runs/

# Download a specific model
gsutil cp gs://mlops-training-stdne/runs/2026-01-22/12-30-45/checkpoints/best_model.pt ./

# Download entire run directory
gsutil -m cp -r gs://mlops-training-stdne/runs/2026-01-22/12-30-45/ ./local_run/
```

### 7.3 Using AIP_MODEL_DIR (Optional)

Vertex AI provides the `AIP_MODEL_DIR` environment variable pointing to a GCS location. To use it, modify your training code:

```python
import os

# In train.py, after training completes:
model_dir = os.environ.get("AIP_MODEL_DIR", "models/")
# Save model to this directory
```

---

## Phase 8: Troubleshooting

### Common Issues

**1. DVC-GS Module Not Found**

If you see `ERROR: unexpected error - gs is supported, but requires 'dvc-gs' to be installed`:

```bash
# Ensure dependencies are synced
uv sync

# Explicitly install dvc-gs package
uv add "dvc-gs>=3.0.2"
uv sync

# Verify installation
uv run python -c "import dvc_gs; print('dvc-gs installed successfully')"

# IMPORTANT: Use 'python -m dvc' instead of 'dvc' to ensure correct environment
uv run python -m dvc pull -r storage
```

**Why this happens**: The `dvc` command might use a different Python interpreter than `uv run python`. Using `python -m dvc` ensures DVC runs in the same environment where `dvc-gs` is installed.

**2. Permission Denied on GCS Bucket**
```bash
# Ensure your service account has Storage Object Admin role
gcloud projects add-iam-policy-binding mlops-485010 \
    --member="serviceAccount:[YOUR_SERVICE_ACCOUNT]" \
    --role="roles/storage.objectAdmin"

# Also ensure application default credentials are set
gcloud auth application-default login
```

**3. Docker Build Fails on Apple Silicon Mac**

If you see `InvalidBaseImagePlatform` or Python interpreter errors when building:

```bash
# Always build for linux/amd64 platform (Vertex AI runs on x86_64)
docker build --platform linux/amd64 -f dockerfiles/vertex_train.dockerfile -t ${IMAGE_URI} .
```

The PyTorch CUDA images are only available for linux/amd64, not ARM64.

**4. GPU Quota Exceeded**
```bash
# Check your GPU quota
gcloud compute regions describe europe-west1 --format="json" | jq '.quotas[] | select(.metric | contains("GPU"))'

# Request quota increase in GCP Console:
# IAM & Admin > Quotas > Filter by "GPU"
```

**5. Image Pull Failed**
```bash
# Verify image exists
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}

# Re-authenticate if needed
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

**6. Data Not Found at /gcs/ Path**
- Ensure the bucket name is correct (no `gs://` prefix in path)
- Check bucket permissions
- Verify data exists: `gsutil ls gs://mlops-training-stdne/processed/`
- Make sure you uploaded processed data (not just DVC cache)

**7. Out of Memory (OOM)**
- Reduce `train.batch_size` in the config
- Use a machine with more memory
- Enable gradient checkpointing in your model

### Useful Commands

```bash
# Check job status
gcloud ai custom-jobs list --region=europe-west1 --filter="state=JOB_STATE_RUNNING"

# Cancel a running job
gcloud ai custom-jobs cancel [JOB_ID] --region=europe-west1

# View logs from a completed job
gcloud logging read "resource.labels.job_id=[JOB_ID]" --limit=100

# Check available machine types
gcloud compute machine-types list --filter="zone:europe-west1-b"

# Check available GPU types in region
gcloud compute accelerator-types list --filter="zone:europe-west1-b"
```

---

## Quick Reference: Complete Workflow

```bash
# ============================================
# FIRST: Set your variables
# ============================================
export PROJECT_ID=mlops-485010
export BUCKET_NAME=mlops-training-stdne  # e.g., mlops-training-myname
export REGION=europe-west1

# ============================================
# 1. GCP Setup (one-time)
# ============================================
gcloud auth login
gcloud auth application-default login
gcloud config set project ${PROJECT_ID}
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# Create your bucket
gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION} --uniform-bucket-level-access

# Create artifact registry
gcloud artifacts repositories create mlops-training --repository-format=docker --location=${REGION}
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# ============================================
# 2. DVC Setup (one-time)
# ============================================
# Add your bucket as DVC remote
uv run python -m dvc remote add trainremote gs://${BUCKET_NAME}
uv run python -m dvc remote default trainremote

# Ensure dependencies are synced (especially dvc-gs)
uv sync

# Ensure dvc-gs is installed
uv add "dvc-gs>=3.0.2"
uv sync

# Pull data from source bucket (use python -m dvc for correct environment)
uv run python -m dvc pull -r storage

# Push to your bucket
uv run python -m dvc push -r trainremote

# ============================================
# 3. Upload CSV files to GCS (images already in bucket from DVC)
# ============================================
gsutil cp data/raw/train.csv gs://${BUCKET_NAME}/raw/
gsutil cp data/raw/test.csv gs://${BUCKET_NAME}/raw/

# ============================================
# 3b. Run preprocessing on Vertex AI (RECOMMENDED)
# ============================================
# Update preprocessing config
# Config already has correct project ID (mlops-485010)
# sed -i "s/\[YOUR_PROJECT_ID\]/${PROJECT_ID}/g" configs/vertex_preprocess_config.yaml

# Submit preprocessing job
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="preprocessing_$(date +%Y%m%d_%H%M%S)" \
    --config=configs/vertex_preprocess_config.yaml

# Wait for completion and verify
gsutil ls gs://${BUCKET_NAME}/processed/

# ============================================
# 3c. Alternative: Preprocess locally (not recommended for large datasets)
# ============================================
# uv run preprocess
# gsutil -m cp -r data/processed/* gs://${BUCKET_NAME}/processed/

# ============================================
# 4. Build and push containers
# ============================================
# Training image
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-training/mlops-trainer:v1"
docker build --platform linux/amd64 -f dockerfiles/vertex_train.dockerfile -t ${IMAGE_URI} .
docker push ${IMAGE_URI}

# Preprocessing image (optional but recommended)
export PREPROCESS_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-training/mlops-preprocessor:v1"
docker build --platform linux/amd64 -f dockerfiles/vertex_preprocess.dockerfile -t ${PREPROCESS_IMAGE_URI} .
docker push ${PREPROCESS_IMAGE_URI}

# ============================================
# 5. Update config with your values
# ============================================
# Config already has correct project ID (mlops-485010)
# sed -i "s/\[YOUR_PROJECT_ID\]/${PROJECT_ID}/g" configs/vertex_train_config.yaml
sed -i "s/\[YOUR_BUCKET_NAME\]/${BUCKET_NAME}/g" configs/vertex_train_config.yaml

# ============================================
# 6. Submit training job
# ============================================
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="mlops_training_$(date +%Y%m%d_%H%M%S)" \
    --config=configs/vertex_train_config.yaml

# ============================================
# 7. Monitor training
# ============================================
gcloud ai custom-jobs list --region=${REGION}

# ============================================
# 8. Run evaluation
# ============================================
# Update eval.model_name in configs/vertex_eval_config.yaml with your run timestamp
# Then submit evaluation job
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name="evaluation_$(date +%Y%m%d_%H%M%S)" \
    --config=configs/vertex_eval_config.yaml

# Check evaluation metrics
gsutil cat gs://${BUCKET_NAME}/runs/[DATE]/[TIME]/eval_metrics.csv

# ============================================
# 9. Download results
# ============================================
gsutil -m cp -r gs://${BUCKET_NAME}/runs/[DATE]/[TIME]/ ./results/
```

---

## Cost Estimation

| Resource | Hourly Cost (approx.) | Typical Job Duration | Cost per Run |
|----------|----------------------|---------------------|--------------|
| **Preprocessing (n1-standard-4)** | ~$0.19 | 5-10 min | **~$0.03-0.05** |
| **Training (n1-standard-8 + T4)** | ~$0.73 | 30-60 min | **~$0.35-0.75** |
| NVIDIA V100 upgrade | ~$2.48 | 15-30 min | ~$0.60-1.25 |
| NVIDIA A100 upgrade | ~$4.00 | 10-20 min | ~$0.65-1.35 |
| Storage (per GB/month) | ~$0.02 | - | ~$0.50/month for 25GB |

**Cost Comparison: Preprocessing**

| Approach | Data Transfer | Compute | Total Time | Total Cost |
|----------|--------------|---------|------------|------------|
| **Local** | Download 11GB + Upload 2GB | Free | 30-60 min | $0 (but uses your time/bandwidth) |
| **Vertex AI** | None (GCS → GCS) | ~$0.05 | 5-10 min | **~$0.05** |

**Tip**: Use preemptible/spot VMs for ~70% cost reduction (add `scheduling.strategy: SPOT` to your config).

**Why Vertex AI preprocessing is worth it:**
- Saves 20-50 minutes of your time
- No local storage needed (saves ~13GB)
- Reproducible across team
- Cost: ~$0.05 (less than a coffee)

---

## Next Steps

1. [ ] Complete Phase 1: GCP setup and authentication
2. [ ] Create your own GCS bucket: `gs://mlops-training-stdne`
3. [ ] Configure DVC with your bucket as a new remote
4. [ ] Pull data from source: `uv run python -m dvc pull -r storage`
5. [ ] Push data to your bucket: `uv run python -m dvc push -r trainremote`
6. [ ] Upload CSV files to GCS: `gsutil cp data/raw/*.csv gs://mlops-training-stdne/raw/`
7. [ ] Update `dockerfiles/vertex_train.dockerfile`
8. [ ] Build and push training Docker image (`mlops-trainer:v1`)
9. [ ] (Optional) Build and push preprocessing Docker image (`mlops-preprocessor:v1`)
10. [ ] Create `configs/vertex_preprocess_config.yaml` and `configs/vertex_train_config.yaml`
11. [ ] Update preprocessing config to use `mlops-preprocessor:v1` if you built it
10. [ ] Submit preprocessing job on Vertex AI (RECOMMENDED)
11. [ ] Verify processed data in GCS
12. [ ] Submit training job
13. [ ] Monitor training progress
14. [ ] Submit evaluation job after training completes
15. [ ] Review evaluation metrics
16. [ ] Download model and results
