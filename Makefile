SHELL := /bin/bash

env_raw = $(shell awk '/^[[:space:]]*$(1)=/ { line=$$0; sub(/^[[:space:]]*$(1)=/, "", line); val=line } END { print val }' .env 2>/dev/null)

UV_ENV := $(shell bash scripts/expand_user_path.sh "$(call env_raw,UV_PROJECT_ENVIRONMENT)")
ifeq ($(UV_ENV),)
UV_ENV := .venv
endif
export UV_PROJECT_ENVIRONMENT := $(UV_ENV)
export VIRTUAL_ENV :=

DQT_DATA := $(shell bash scripts/expand_user_path.sh "$(call env_raw,DQT_DATA_DIR)")
ifeq ($(DQT_DATA),)
DQT_DATA := data
endif
export DQT_DATA_DIR := $(DQT_DATA)

START_DATE := $(shell bash scripts/expand_user_path.sh "$(call env_raw,START_DATE)")
ifeq ($(START_DATE),)
START_DATE := 2024-07-01
endif
export START_DATE

SMOKE_START := $(shell python3 -c "from datetime import date, timedelta; print((date.today() - timedelta(days=14)).isoformat())")

.DEFAULT_GOAL := help
.PHONY: help install lint lint-sql test features hybrid sarima-wf sarima-publish shadow-dqt guardrails-assess smoke

help: ## list targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## uv venv + Jupyter kernel
	bash scripts/install.sh

lint-sql: ## sqlfluff over SQL/
	uv run sqlfluff lint SQL/

lint: ## ruff + sqlfluff
	uv run ruff check src/ scripts/ --fix
	$(MAKE) lint-sql

test: ## pytest
	uv run pytest tests/ -q

features: ## Snowflake → features.parquet  [FORCE=1]
	@echo "DQT_DATA_DIR=$(DQT_DATA_DIR)  START_DATE=$(START_DATE)"
	uv run python scripts/build_features.py $(if $(FORCE),--force,)

hybrid: ## hybrid_quantiles.parquet  [FORCE=1]
	uv run python scripts/build_hybrid.py $(if $(FORCE),--force,)

sarima-wf: ## walk-forward SARIMA dial  [FORCE=1 | APPEND=1]
	uv run python scripts/build_sarima_walkforward.py $(if $(FORCE),--force,) $(if $(APPEND),--append,)

sarima-publish: ## kill-switch + schedule
	uv run python scripts/publish_sarima_dial.py --data-dir $(DQT_DATA_DIR)

shadow-dqt: ## daily shadow eval  [SKIP_SF=1 | REFRESH=1]
	uv run python scripts/shadow_dqt_daily.py --data-dir $(DQT_DATA_DIR) \
		$(if $(SKIP_SF),--skip-sf,) $(if $(REFRESH),--refresh,)

guardrails-assess: ## guardrail fire-rates  [QUICK=1]
	uv run python scripts/assess_dqt_guardrails.py --data-dir $(DQT_DATA_DIR) \
		$(if $(QUICK),--quick,)

smoke: ## 14-day features pull → data/current
	DQT_DATA_DIR=data/current START_DATE=$(SMOKE_START) \
		uv run python scripts/build_features.py --force --data-dir data/current --start $(SMOKE_START)
