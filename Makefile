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
	uv run python -m pytest -q

eval: ## Live functional eval: re-record ALL cases (needs keys + $). Prefer record-missing.
	LIVE_LLM=1 uv run python -m pytest evals/test_functional.py -v

record-missing: ## Cheap re-record: call the model ONLY for cache misses, replay the rest (needs keys)
	LIVE_LLM=1 RECORD_MISSING=1 uv run python -m pytest evals/test_functional.py -v

eval-ci: ## Offline functional eval: replay recorded inputs AND recorded judge verdicts (keyless)
	uv run python -m pytest evals/test_functional.py -v

redteam: ## Adversarial red-team: replay recorded agent responses, grade live -> graded report
	uv run python -m adversarial.run

redteam-live: ## Adversarial red-team: re-record ALL cases (needs keys + $). Prefer redteam-record-missing.
	LIVE_LLM=1 uv run python -m adversarial.run

redteam-record-missing: ## Cheap red-team re-record: model calls only for misses (needs keys)
	LIVE_LLM=1 RECORD_MISSING=1 uv run python -m adversarial.run

agent-tests: ## Agent-reliability suite: tests the graph (tool calls, loop safety, state, recovery). Keyless.
	uv run python -m pytest agent_tests/ -v

meta-eval: ## Challenge the judge: replay cached judge scores, report agreement vs human gold set
	uv run python -m meta_eval.run

meta-eval-live: ## Meta-eval LIVE: real Claude judge over the gold set, re-records scores (needs ANTHROPIC key)
	LIVE_LLM=1 uv run python -m meta_eval.run --live

history: ## Show the eval metrics-over-time trend (drift made visible)
	uv run python -m evals.history

record-history: ## Run the suite once and append a summary row to evals/history/ (needs ANTHROPIC key)
	uv run python -m evals.record_history
