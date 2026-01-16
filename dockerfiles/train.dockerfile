# 1. Use -bookworm-slim instead of -alpine
# This supports PyTorch wheels (manylinux)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# 2. Install git (Required for DVC to work)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

# Important: Copy .dvc and .git folders so 'dvc pull' knows where to get data
COPY .dvc .dvc
COPY .dvcignore .dvcignore

ENV DVC_NO_SCM=true


COPY data data/
COPY configs configs/
COPY src src/
COPY README.md README.md

RUN uv sync --frozen

ENTRYPOINT ["/bin/sh", "-c"]

# Default command (can be overridden by Vertex AI)
# This chains the pull AND the training
CMD ["uv run dvc pull && uv run train"]