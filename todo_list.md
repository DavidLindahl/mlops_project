# DTU MLOps Final Project — Implementation TODO

Living checklist for the DTU MLOps final project. Organized in chronological order with **FIRST**, **LATER**, and **CONTINUOUS** sections.

**Legend**
- **MUST** = minimum for a solid/gradeable MLOps project
- **OPT** = nice-to-have / stretch goal
- **Depends on** = do this earlier (or expect pain)

---

## FIRST

### Project framing
- [ ] **MUST** Define problem, dataset, evaluation metric, and a “good enough” baseline.
- [ ] **MUST** Define the end-to-end demo path: `train → model artifact → serve inference → observe/monitor`.
- [ ] **MUST** Assign ownership (data, training, API, infra, report).
- [ ] **MUST** Create a shared backlog (GitHub Issues/Projects) and copy this checklist there.
- [ ] **OPT** Define acceptance criteria for milestones (MVP local, deployed, monitored).
- [ ] **OPT** Draw an initial architecture diagram (boxes + arrows).

### Git

#### Repo setup
- [ ] **MUST** Create repo + ensure all members have write access.
- [ ] **MUST** Protect `main` (PR required + required checks).
- [ ] **MUST** Add `.gitignore` (Python + data/artifacts + secrets).
- [ ] **MUST** Add a minimal `README.md` (how to run train, how to run API, how to run tests).
- [ ] **OPT** Add `CODEOWNERS`, PR template, issue templates.

#### Commit/branch conventions
- [ ] **MUST** Use feature branches + PRs; keep `main` always runnable.
- [ ] **OPT** Adopt a commit convention (e.g., Conventional Commits).
- [ ] **OPT** Decide branch naming convention (e.g., `feat/...`, `fix/...`, `chore/...`).

#### Pre-commit (depends on: formatter/linter chosen)
- [ ] **MUST** Add `pre-commit` with formatter/linter hooks (at minimum `ruff`).
- [ ] **MUST** Run pre-commit in CI too (so local + CI agree).
- [ ] **OPT** Add hooks: type-checking (`mypy`), YAML check, trailing whitespace, etc.

### Environment (uv)

#### Dependency & Python pinning
- [ ] **MUST** Use `uv` with `pyproject.toml` as the single source of dependencies.
- [ ] **MUST** Commit the lockfile (`uv.lock`) for reproducible installs.
- [ ] **MUST** Pin a Python version (document it; optionally add `.python-version`).
- [ ] **OPT** Add a small “quickstart” block in README: `uv sync`, `uv run pytest`, `uv run <cmd>`.

#### Common commands
- [ ] **MUST** Add one command entry point per workflow: `format`, `lint`, `test`, `train`, `serve`.
  - Example: `Makefile` / `justfile` / `taskfile.yml` / `scripts/`.
- [ ] **OPT** Standardize `.env` usage for local secrets (never commit `.env`).

### Repo structure & template (depends on: Git repo setup)
- [ ] **MUST** Create a clean project structure (template/cookiecutter is fine).
- [ ] **MUST** Ensure `reports/README.md` exists in the expected path (course tooling).
- [ ] **MUST** Put library code under `src/<package_name>/...` and keep scripts thin.
- [ ] **OPT** Package the project via `pyproject.toml` (editable install for dev).
- [ ] **OPT** Add console scripts entry points (e.g., `train`, `serve`) in `pyproject.toml`.

### Data pipeline
- [ ] **MUST** Implement data ingestion + preprocessing (write outputs to `data/processed/` or similar).
- [ ] **MUST** Make it deterministic (fixed seeds, stable splits, no hidden randomness).
- [ ] **MUST** Record dataset source/version (in code + README).
- [ ] **OPT** Add basic data validation (schema/ranges/missingness) and fail fast.

### Model + training loop
- [ ] **MUST** Implement `model.py` and `train.py` and run end-to-end locally.
- [ ] **MUST** Save a trained artifact (e.g., `.pt`/`.pkl`) + metrics summary (`.json`).
- [ ] **MUST** Make training reproducible (seed + deterministic ops where possible).
- [ ] **OPT** Add `predict.py` (batch inference) and/or `evaluate.py`.

### Config management (Hydra + YAML) (depends on: training loop exists)
- [ ] **MUST** Add YAML configs for experiments (paths, model params, train params).
- [ ] **MUST** Use Hydra to load configs and manage overrides.
- [ ] **OPT** Save the resolved config next to each run artifact.
- [ ] **OPT** Add Hydra sweeps (grid/random) for quick comparisons.

### Logging & experiment tracking (depends on: training loop + configs exist)
- [ ] **MUST** Add Python `logging` for key events (start/end, data version, metrics).
- [ ] **MUST** Log hyperparameters + metrics (at least JSON locally).
- [ ] **OPT** Add W&B (metrics + artifacts + config logging).
- [ ] **OPT** Log extra artifacts (model file, plots, example predictions).

### Data version control (DVC or equivalent) (depends on: data pipeline exists)
- [ ] **MUST** Track at least one data artifact with DVC (raw or processed).
- [ ] **MUST** Ensure large data is not in git history.
- [ ] **OPT** Configure a remote early (later: cloud bucket) and make `dvc pull` work.

### Docker (depends on: local train/serve commands work)
- [ ] **MUST** Create Dockerfile(s) for at least one component (API or training).
- [ ] **MUST** Build and run locally; document exact commands.
- [ ] **OPT** Multi-stage build, smaller images, non-root user.

---

## LATER

### CI (GitHub Actions) (depends on: tests exist)
- [ ] **MUST** CI workflow runs unit tests on PR/push.
- [ ] **MUST** CI workflow enforces lint/format (or runs `pre-commit`).
- [ ] **MUST** CI prints coverage summary (or uploads an artifact/report).
- [ ] **OPT** Add `uv` caching to speed up CI.
- [ ] **OPT** Matrix testing (multiple Python versions / OS if useful).
- [ ] **OPT** Build Docker image in CI and push to a registry.

### Cloud setup
- [ ] **MUST** Create/choose GCP project; confirm billing/credits; set budgets/alerts.
- [ ] **MUST** Enable required APIs (Storage, Artifact Registry, Cloud Run/Functions, etc.).
- [ ] **MUST** Create service accounts with minimal IAM permissions.
- [ ] **MUST** Document credential handling (local dev vs CI vs cloud runtime).
- [ ] **OPT** Prefer Workload Identity / avoid long-lived keys.

### Using the cloud (depends on: Docker build works locally)
- [ ] **MUST** Create cloud storage bucket(s) for data and/or artifacts.
- [ ] **MUST** Connect DVC remote to cloud storage (or document a `gsutil` download path).
- [ ] **MUST** Set up Artifact Registry (container images).
- [ ] **MUST** Store secrets in Secret Manager (tokens/config) if needed.
- [ ] **OPT** Automate image build triggers (CI → registry).
- [ ] **OPT** Run training in the cloud (Compute Engine / managed training) using your container.

### API (FastAPI) (depends on: model artifact exists + inference works locally)
- [ ] **MUST** Build a FastAPI app exposing `/predict` (loads model artifact).
- [ ] **MUST** Add input validation (pydantic) + clear error messages.
- [ ] **MUST** Provide example request(s) in README (`curl` + Python).
- [ ] **OPT** Add health endpoint (`/health`) and versioned routes (`/v1/predict`).

### Cloud deployment
- [ ] **MUST** Deploy API to cloud (e.g., Cloud Run).
- [ ] **MUST** Prove end-to-end behavior (cloud endpoint returns predictions).
- [ ] **OPT** Add continuous deployment on merge to `main`.
- [ ] **OPT** Add rollout strategy (traffic splitting / canary) if you change models often.

### API testing (depends on: API exists; cloud tests depend on: deployment exists)
- [ ] **MUST** Write functional API tests (status codes + schema + known example).
- [ ] **MUST** Run API tests in CI (local container; optionally also against deployed endpoint).
- [ ] **OPT** Contract test to catch breaking changes.
- [ ] **OPT** Load test and document latency/throughput.

### Monitoring & observability (depends on: deployed API)
- [ ] **MUST** Log requests/errors with enough context to debug (avoid PII).
- [ ] **MUST** Add basic metrics: request count, latency, error rate.
- [ ] **MUST** Create dashboards and at least one alert (high error rate/latency/budget).
- [ ] **OPT** Structured logs + request IDs; traces if you have time.

### Drift detection (depends on: deployed API + storage of requests/predictions)
- [ ] **MUST** Store inputs/outputs from production (CSV/DB/bucket).
- [ ] **MUST** Run a drift report comparing reference vs recent data (e.g., Evidently).
- [ ] **MUST** Define a retraining trigger rule (even if simple).
- [ ] **OPT** Automate drift checks (scheduled job).
- [ ] **OPT** Deploy drift reporting as a service/API.

### Scaling (only if relevant)
- [ ] **OPT** Optimize data loading if it is a bottleneck.
- [ ] **OPT** Distributed training only if training time is genuinely limiting.
- [ ] **OPT** Improve inference speed (e.g., ONNX/quantization) and document tradeoffs.

### Report + deliverables (start early, finish late)
- [ ] **MUST** Fill `reports/README.md` using the course template requirements.
- [ ] **MUST** Keep repo root `README.md` and required paths in place.
- [ ] **MUST** Include an architecture diagram of your pipeline.
- [ ] **MUST** Ensure every group member can explain each system component.
- [ ] **OPT** Publish docs via GitHub Pages.

---

## CONTINUOUS

### Testing & quality gates
- [ ] **MUST** Unit tests for data code (download/parsing/preprocess invariants).
- [ ] **MUST** Unit/smoke tests for model construction and a tiny training step.
- [ ] **MUST** Track coverage (don’t obsess; just keep it from collapsing).
- [ ] **OPT** Integration test: “train for 1 epoch on tiny data and produce artifact”.
- [ ] **OPT** Property-based tests for data assumptions.

### Code review & repo health
- [ ] **MUST** Small PRs, clear descriptions, and keep `main` green.
- [ ] **MUST** Update README when commands/paths change.
- [ ] **OPT** Keep a CHANGELOG (or at least tag “milestone” releases).

### Reproducibility discipline
- [ ] **MUST** Don’t merge changes that break `uv sync` + `uv run test/train/serve`.
- [ ] **MUST** Keep seeds/config/artifacts consistent and traceable.
- [ ] **OPT** Track model/data lineage more formally (e.g., W&B artifacts).

---

## Suggested Definition of Done (quick sanity check)

- [ ] **MUST** `uv run train` works locally and produces a model artifact.
- [ ] **MUST** `uv run serve` (or `uv run python -m <pkg>.api`) works locally and returns predictions.
- [ ] **MUST** Docker image builds and runs the API locally.
- [ ] **MUST** CI runs tests + lint and is required for merging to `main`.
- [ ] **MUST** Cloud endpoint works and has at least basic automated API tests.
- [ ] **MUST** Logs + metrics exist, and at least one alert can fire.
- [ ] **MUST** Drift check can be executed (manual is fine; automated is better).
