# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/mlops_project/evaluate.py"] # Simply need to add arguments that evaluate.py has. Can be done via .yaml file
