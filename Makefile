# GCP Vertex AI Automation
# Usage: make <target> [VARIABLE=value]

PROJECT_ID ?= mlops-485010
REGION ?= europe-west1
REPO_NAME ?= mlops-training
BUCKET ?= gs://mlops-training-stdne
TAG ?= v1

# Docker image URIs
TRAIN_IMAGE = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/mlops-trainer:$(TAG)
PREPROCESS_IMAGE = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/mlops-preprocessor:$(TAG)
# API uses same image as training (MLOps best practice: one image, multiple uses)
API_IMAGE = $(TRAIN_IMAGE)

# API deployment settings
MODEL_VERSION ?= v1
ENDPOINT_NAME ?= mlops-api-endpoint
MODEL_NAME ?= mlops-api-model

.PHONY: help build-train build-preprocess push-train push-preprocess submit-train submit-preprocess promote-model list-runs deploy-api

help:
	@echo "GCP Vertex AI Commands:"
	@echo "  make build-train [TAG=v1]          - Build training Docker image"
	@echo "  make build-preprocess [TAG=v1]      - Build preprocessing Docker image"
	@echo "  make push-train [TAG=v1]            - Build and push training image (also used for API)"
	@echo "  make push-preprocess [TAG=v1]       - Build and push preprocessing image"
	@echo "  make submit-train                   - Submit training job"
	@echo "  make submit-preprocess               - Submit preprocessing job"
	@echo "  make promote-model [RUN=<path>] [VERSION=<v1>]  - Promote model (auto-finds latest if RUN not specified)"
	@echo "  make list-runs [DATE=2026-01-23]     - List available training runs (optionally filter by date)"
	@echo "  make deploy-api [MODEL_VERSION=v1]  - Deploy API to Vertex AI Endpoints"

build-train:
	docker build --platform linux/amd64 \
		-f dockerfiles/vertex_train.dockerfile \
		-t $(TRAIN_IMAGE) .

build-preprocess:
	docker build --platform linux/amd64 \
		-f dockerfiles/vertex_preprocess.dockerfile \
		-t $(PREPROCESS_IMAGE) .

push-train: build-train
	docker push $(TRAIN_IMAGE)
	@echo "✅ Image pushed: $(TRAIN_IMAGE)"
	@echo "   (Same image used for training and API)"

push-preprocess: build-preprocess
	docker push $(PREPROCESS_IMAGE)
	@echo "✅ Image pushed: $(PREPROCESS_IMAGE)"

submit-train:
	gcloud ai custom-jobs create \
		--region=$(REGION) \
		--display-name="mlops_training_$$(date +%Y%m%d_%H%M%S)" \
		--config=configs/vertex_train_config.yaml \
		--project=$(PROJECT_ID)

submit-preprocess:
	gcloud ai custom-jobs create \
		--region=$(REGION) \
		--display-name="mlops_preprocess_$$(date +%Y%m%d_%H%M%S)" \
		--config=configs/vertex_preprocess_config.yaml \
		--project=$(PROJECT_ID)

promote-model:
	@if [ -z "$(RUN)" ]; then \
		echo "🔍 Finding latest run..."; \
		LATEST_DATE=$$(gsutil ls $(BUCKET)/runs/ | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}/$$' | sort -r | head -1 | xargs basename); \
		if [ -z "$$LATEST_DATE" ]; then \
			echo "❌ No runs found in $(BUCKET)/runs/"; \
			exit 1; \
		fi; \
		LATEST_RUN=$$(gsutil ls $(BUCKET)/runs/$$LATEST_DATE/ | grep -v '^$$' | sort -r | head -1 | xargs basename); \
		if [ -z "$$LATEST_RUN" ]; then \
			echo "❌ No runs found for date $$LATEST_DATE"; \
			exit 1; \
		fi; \
		SELECTED_RUN="$$LATEST_DATE/$$LATEST_RUN"; \
		echo "✅ Found latest run: $$SELECTED_RUN"; \
	else \
		SELECTED_RUN="$(RUN)"; \
	fi; \
	VERSION=$${VERSION:-$$(echo $$SELECTED_RUN | cut -d'/' -f1 | tr -d '-')}; \
	echo "📦 Promoting model from $$SELECTED_RUN to version $$VERSION..."; \
	gsutil cp $(BUCKET)/runs/$$SELECTED_RUN/checkpoints/best_model.pt $(BUCKET)/models/$$VERSION/model.pt && \
	gsutil cp $(BUCKET)/models/$$VERSION/model.pt $(BUCKET)/models/latest/model.pt && \
	echo "✅ Model promoted to models/$$VERSION/ and models/latest/"

list-runs:
	@if [ -z "$(DATE)" ]; then \
		echo "📋 Available training runs:"; \
		echo ""; \
		for date_dir in $$(gsutil ls $(BUCKET)/runs/ | grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2}/$$'); do \
			date=$$(basename $$date_dir); \
			echo "📅 $$date:"; \
			gsutil ls $$date_dir 2>/dev/null | grep -v '^$$' | while read run; do \
				run_name=$$(basename $$run); \
				echo "   └─ $$run_name"; \
			done || echo "   (no runs)"; \
			echo ""; \
		done || echo "No runs found"; \
	else \
		echo "📋 Training runs for $(DATE):"; \
		gsutil ls $(BUCKET)/runs/$(DATE)/ 2>/dev/null | grep -v '^$$' | while read run; do \
			run_name=$$(basename $$run); \
			echo "   └─ $$run_name"; \
		done || echo "   (no runs found for $(DATE))"; \
	fi

deploy-api: push-train
	@echo "🚀 Deploying API to Vertex AI Endpoints..."
	@echo "   Endpoint: $(ENDPOINT_NAME)"
	@echo "   Model Version: $(MODEL_VERSION)"
	@echo "   Image: $(API_IMAGE) (reusing training image)"
	@echo ""
	@echo "Step 1: Creating model resource..."
	@MODEL_ID="$(MODEL_NAME)-$$(date +%Y%m%d-%H%M%S)" || MODEL_ID="$(MODEL_NAME)-$$(date +%s)"; \
	gcloud ai models upload \
		--region=$(REGION) \
		--display-name="$$MODEL_ID" \
		--container-image-uri=$(API_IMAGE) \
		--container-command="python,-m,uvicorn,mlops_project.api:app,--host,0.0.0.0,--port,8080" \
		--container-env-vars="MODEL_PATH=/gcs/mlops-training-stdne/models/$(MODEL_VERSION)/model.pt,MODEL_VERSION=$(MODEL_VERSION)" \
		--container-ports=8080 \
		--project=$(PROJECT_ID); \
	echo "✅ Model created: $$MODEL_ID"; \
	echo ""; \
	echo "Step 2: Creating or getting endpoint..."; \
	ENDPOINT_ID=$$(gcloud ai endpoints list --region=$(REGION) --filter="displayName:$(ENDPOINT_NAME)" --format="value(name)" --limit=1 2>/dev/null || echo ""); \
	if [ -z "$$ENDPOINT_ID" ]; then \
		echo "Creating new endpoint..."; \
		ENDPOINT_ID=$$(gcloud ai endpoints create \
			--region=$(REGION) \
			--display-name=$(ENDPOINT_NAME) \
			--project=$(PROJECT_ID) \
			--format="value(name)"); \
		echo "✅ Endpoint created: $$ENDPOINT_ID"; \
	else \
		echo "Using existing endpoint: $$ENDPOINT_ID"; \
	fi; \
	echo ""; \
	echo "Step 3: Deploying model to endpoint..."; \
	gcloud ai endpoints deploy-model $$ENDPOINT_ID \
		--region=$(REGION) \
		--model=$$MODEL_ID \
		--display-name="$$MODEL_ID" \
		--machine-type=n1-standard-4 \
		--accelerator-type=nvidia-tesla-t4 \
		--accelerator-count=1 \
		--min-replica-count=0 \
		--max-replica-count=1 \
		--project=$(PROJECT_ID); \
	echo ""; \
	echo "✅ API deployed successfully!"; \
	echo ""; \
	echo "🌐 Endpoint ID: $$ENDPOINT_ID"; \
	echo "   Prediction URL: https://$$ENDPOINT_ID-prediction.$(REGION)-aiplatform.google.com"; \
	echo ""; \
	echo "To test:"; \
	echo "  export ENDPOINT_ID=$$ENDPOINT_ID"; \
	echo "  export API_URL=https://$$ENDPOINT_ID-prediction.$(REGION)-aiplatform.google.com"; \
	echo "  uv run inference path/to/images/"
