# ============================================================
# Vertex AI Training Container
# Uses NVIDIA CUDA base + Python 3.12 to match project requirements
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml pyproject.toml

# Install PyTorch with CUDA 12.1 support and all other dependencies
# Using uv for fast, reproducible installs
RUN uv pip install --system --no-cache \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && uv pip install --system --no-cache \
    hydra-core>=1.3.2 \
    pandas>=2.3.3 \
    pillow>=12.1.0 \
    tqdm>=4.66.5 \
    timm>=1.0.24 \
    numpy>=1.25.2 \
    omegaconf \
    wandb>=0.24.0 \
    python-dotenv>=1.2.1 \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    requests>=2.31.0

# Copy source code and configs
COPY configs configs/
COPY src src/
COPY README.md README.md

# Install the project as editable package
RUN uv pip install --system --no-deps -e .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Vertex AI calls this entrypoint
ENTRYPOINT ["python", "-m", "mlops_project.train"]