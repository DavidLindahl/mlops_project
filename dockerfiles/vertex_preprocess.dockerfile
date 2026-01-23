# Lighter image for preprocessing - no CUDA needed
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Install dependencies (no PyTorch needed for preprocessing)
RUN uv venv && \
    uv pip install pandas pillow tqdm hydra-core && \
    uv sync --frozen --no-install-project

# Copy source code and configs
COPY configs configs/
COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

# Install the project
RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "preprocess"]