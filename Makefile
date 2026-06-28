# ReadmitRisk: developer & reproducibility entry points.
#
# Local stages run via `uv` (fast, what CI uses). The full containerized stack
# (Synthea one-shot + pipeline + Streamlit demo) runs via `docker compose`.

UV ?= uv
COMPOSE ?= docker compose
PY := $(UV) run
APP := src/readmitrisk/ui/app.py
PORT ?= 8501

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
.PHONY: install
install: ## Create the uv venv and install all deps (incl. dev)
	$(UV) sync --extra dev

.PHONY: lint
lint: ## Ruff lint
	$(PY) ruff check src tests

.PHONY: format
format: ## Ruff format
	$(PY) ruff format src tests

.PHONY: test
test: ## Run the pytest suite (incl. censoring/metric-correctness invariant)
	$(PY) pytest

# --------------------------------------------------------------------------- #
# Pipeline stages (local)
# --------------------------------------------------------------------------- #
.PHONY: generate
generate: ## Generate synthetic EHR (Synthea if available, else deterministic fallback)
	$(PY) readmitrisk generate

.PHONY: generate-synthea
generate-synthea: ## Force the real Synthea backend (needs Java/jar or the synthea container)
	READMIT_GENERATOR=synthea $(PY) readmitrisk generate --backend synthea

.PHONY: cohort
cohort: ## Build the time-to-event cohort via DuckDB SQL
	$(PY) readmitrisk cohort

.PHONY: train
train: ## Fit Cox PH + Random Survival Forest and persist artifacts
	$(PY) readmitrisk train

.PHONY: eval
eval: ## Evaluate survival metrics + enforce the C-index gate (non-zero exit on failure)
	$(PY) readmitrisk eval

.PHONY: fairness
fairness: ## Per-subgroup C-index + calibration audit
	$(PY) readmitrisk fairness

.PHONY: pipeline
pipeline: generate cohort train eval fairness ## Run the full local pipeline end to end

.PHONY: sample
sample: ## Rebuild the small committed cached sample used by tests + CI
	$(PY) python -m readmitrisk.sample_build

# --------------------------------------------------------------------------- #
# Demo + containerized stack
# --------------------------------------------------------------------------- #
.PHONY: demo
demo: ## Launch the Streamlit risk-curve demo locally
	$(PY) streamlit run $(APP) --server.port $(PORT)

.PHONY: up
up: ## docker compose up: generate data + run pipeline + serve the demo
	$(COMPOSE) up --build

.PHONY: up-synthea
up-synthea: ## Same as `up` but generate with the real Synthea container
	$(COMPOSE) --profile synthea up --build

.PHONY: down
down: ## Tear down the stack and remove the data volume
	$(COMPOSE) down -v

.PHONY: clean
clean: ## Remove generated data + artifacts (keeps the committed sample)
	rm -rf data/raw data/cohort data/artifacts data/reports data/*.duckdb
