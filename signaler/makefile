# Makefile for Trading Signal System

.PHONY: help setup install test lint format clean deploy backfill train predict api docker-build docker-up docker-down

# Default target
help:
	@echo "Trading Signal System - Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make deploy        - Deploy to GCP"
	@echo "  make backfill      - Run data backfill"
	@echo "  make train         - Train model"
	@echo "  make predict       - Generate predictions"
	@echo "  make api           - Run API server locally"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"

# Environment setup
setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	cp .env.example .env
	@echo "Setup complete! Don't forget to:"
	@echo "1. Activate venv: source venv/bin/activate"
	@echo "2. Edit .env with your configuration"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

# GCP Deployment
deploy:
	@echo "Deploying to GCP..."
	./scripts/deploy.sh --project-id $(GCP_PROJECT_ID) --region $(GCP_REGION)

deploy-terraform:
	cd terraform && terraform init
	cd terraform && terraform plan -var="project_id=$(GCP_PROJECT_ID)"
	cd terraform && terraform apply -var="project_id=$(GCP_PROJECT_ID)" -auto-approve

# Data operations
backfill:
	python -m src.jobs.backfill_job \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-types all

backfill-ohlcv:
	python -m src.jobs.backfill_job \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--data-types ohlcv

daily-update:
	python -m src.jobs.daily_ingestion --date $(DATE)

# Model operations
train:
	python -m src.jobs.training_job \
		--end-date $(END_DATE) \
		--lookback-months 24

train-validate:
	python -m src.jobs.training_job \
		--validate-only

predict:
	python -m src.training.prediction_pipeline \
		--model-version latest \
		--prediction-date $(DATE)

# API operations
api:
	uvicorn src.api.prediction_service:app --reload --host 0.0.0.0 --port 8080

api-prod:
	gunicorn src.api.prediction_service:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080

# Docker operations
docker-build:
	docker build -f docker/Dockerfile.base -t trading-system/base:latest .
	docker build -f docker/Dockerfile.ingestion -t trading-system/ingestion:latest .
	docker build -f docker/Dockerfile.api -t trading-system/api:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# BigQuery operations
bq-init:
	python scripts/init_bigquery.py

bq-validate:
	python scripts/validate_data.py

# MLflow operations
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Monitoring
monitor-start:
	docker-compose up -d prometheus grafana

monitor-stop:
	docker-compose stop prometheus grafana

# Development helpers
notebook:
	jupyter lab --notebook-dir=notebooks/

docs:
	mkdocs serve

# CI/CD helpers
ci-test:
	pytest tests/ -v --cov=src --cov-report=xml

ci-lint:
	flake8 src/ tests/ --exit-zero
	mypy src/ --ignore-missing-imports

# Variables with defaults
GCP_PROJECT_ID ?= your-project-id
GCP_REGION ?= us-central1
START_DATE ?= 2022-01-01
END_DATE ?= 2023-12-31
DATE ?= $(shell date +%Y-%m-%d)