SHELL := /bin/bash
PY := .venv/bin/python
RUFF := .venv/bin/ruff
UV := uv

.PHONY: help install ingest train serve pipeline test cov lint format check \
        docker-build docker-up docker-down docker-init docker-logs clean

help:
	@echo "FleetIQ targets"
	@echo "  make install       create .venv and install runtime + dev dependencies"
	@echo "  make ingest        load data/raw CSVs into the warehouse"
	@echo "  make train         train, evaluate and register both models"
	@echo "  make serve         run the API and dashboard with autoreload"
	@echo "  make pipeline      run the Prefect refresh flow once"
	@echo "  make test          run pytest"
	@echo "  make cov           run pytest with coverage"
	@echo "  make lint          ruff check"
	@echo "  make format        ruff format"
	@echo "  make check         lint + test"
	@echo "  make docker-up     build and start Postgres, MLflow and the API"
	@echo "  make docker-init   ingest + train inside the Compose stack"
	@echo "  make docker-down   stop the stack"

install:
	$(UV) venv --python 3.13 .venv
	$(UV) pip install --python $(PY) -r requirements-dev.txt

ingest:
	$(PY) -m src.cli ingest --force

train:
	$(PY) -m src.cli train --model all

serve:
	$(PY) -m src.cli serve --reload

pipeline:
	$(PY) -m src.cli pipeline --flow refresh

test:
	$(PY) -m pytest tests -q

cov:
	$(PY) -m pytest tests -q --cov=src --cov=api --cov-report=term-missing

lint:
	$(RUFF) check src api tests

format:
	$(RUFF) format src api tests
	$(RUFF) check src api tests --fix

check: lint test

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-init:
	docker compose run --rm init

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api mlflow

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
