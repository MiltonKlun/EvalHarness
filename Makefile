# LLM Eval Harness — task runner.
# Uses uv for env + execution. `uv run` executes inside the project venv (.venv).
# More targets (eval, eval-ci, redteam, agent-tests, meta-eval) are added per phase.

.DEFAULT_GOAL := help

.PHONY: help install lint fmt test

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install all deps (incl. dev)
	uv venv
	uv pip install -e ".[dev]"

lint: ## Lint with ruff (no changes)
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-format + autofix with ruff
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run the test suite (offline; no API keys needed)
	uv run pytest -q
