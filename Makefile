# =============================================================================
# Shortcuts. Every target is a command you could type yourself -- the Makefile
# is a reminder, not a layer.
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help install env lint format typecheck test test-integration coverage check \
        contract simulate features screen train challenger promote models runs \
        score monitor cycle serve doctor reset notebooks mlflow \
        up down clean-volumes logs psql docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Development -------------------------------------------------------------

install: ## Install the package and its development dependencies
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

env: ## Create .env from the example if it does not exist
	@test -f .env || (cp .env.example .env && echo "created .env -- edit it before running")

lint: ## Run ruff (lint + format check)
	python -m ruff check .
	python -m ruff format --check .

format: ## Apply ruff formatting and safe fixes
	python -m ruff check --fix .
	python -m ruff format .

typecheck: ## Run mypy on src/
	python -m mypy

test: ## Run the unit tests (no database needed)
	python -m pytest tests/unit -v

test-integration: ## Run the full suite (requires a live PostgreSQL)
	POLARIS_RUN_INTEGRATION=1 python -m pytest tests -v

coverage: ## Full suite with a coverage report
	POLARIS_RUN_INTEGRATION=1 python -m pytest tests --cov --cov-report=term-missing

contract: ## Check the feature contract on its own
	python -m pytest tests/unit/test_contract.py -v

check: lint typecheck test ## Everything CI runs, minus the database

# --- The lifecycle -----------------------------------------------------------

doctor: ## Check configuration, database, feature store and registry
	polaris doctor

simulate: ## Generate the simulated Vertex Systems business
	polaris simulate

features: ## Compute point-in-time features for every reference date
	polaris build-features

screen: ## Run the leakage screens over the training data
	polaris screen

train: ## Train the baseline (logistic regression) and try to ship it
	polaris train --algorithm logistic --promote

challenger: ## Train a gradient boosting challenger, without promoting it
	polaris train --algorithm gradient_boosting

promote: ## Promote a version through the gate: make promote V=2
	polaris promote $(V)

models: ## List the registered versions and what is serving
	polaris models

runs: ## Show the training run history
	polaris runs

score: ## Score accounts and record the predictions
	polaris score --explain

monitor: ## Drift now, live performance once the labels arrive
	polaris monitor

cycle: ## Train, register, promote if allowed, and score
	polaris cycle

serve: ## Run the scoring API
	polaris serve --host 0.0.0.0 --port 8000

mlflow: ## Open the MLflow UI on the local tracking store
	mlflow ui --backend-store-uri $${POLARIS_MLFLOW_TRACKING_URI:-sqlite:///mlflow.db} --port 5000

notebooks: ## Re-execute the three notebooks in place
	# nbconvert runs each notebook with its own directory as the working
	# directory, where there is no .env to read -- so the settings are exported
	# here instead of being discovered.
	set -a; test -f .env && . ./.env; set +a; \
	python -m jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_exploration.ipynb \
		notebooks/02_model_selection.ipynb \
		notebooks/03_error_analysis.ipynb

reset: ## Drop the schemas (asks first)
	polaris reset

# --- Docker ------------------------------------------------------------------

up: env ## Start the stack: database, one full cycle, then the API
	docker compose up --build

down: ## Stop the stack (keeps the database volume)
	docker compose down

clean-volumes: ## Stop the stack AND delete the database volume
	docker compose down -v

logs: ## Follow the container logs
	docker compose logs -f

psql: ## Open a psql shell on the compose database
	docker compose exec postgres psql -U $${POLARIS_DB_USER:-polaris_app} -d $${POLARIS_DB_NAME:-polaris}

docker-build: ## Build the application image only
	docker build -t ml-production-platform:local .

# --- Housekeeping ------------------------------------------------------------

clean: ## Remove caches and generated artefacts (keeps .env and the notebooks)
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml junit-*.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf data/artifacts/* data/reports/*
