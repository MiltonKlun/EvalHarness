"""Parametrized functional eval suite — one pytest case per dataset row (plan 2.4).

Behaviour by mode:
  - LIVE_LLM=1  -> real Gemini + Claude calls, recorded to evals/cache. This is `make eval`.
  - default     -> replay recorded Gemini answers; the *metrics still run live*. This is
                   `make eval-ci`. If a case has no recording yet AND no keys, it SKIPS
                   (so the plain `make test` stays green before a baseline is recorded).

A case PASSES only if every metric passes. A JUDGE_ERROR fails the case loudly (never a
silent pass).
"""

from __future__ import annotations

import os

import pytest

from evals.dataset import load_cases
from evals.metrics import Outcome
from evals.runner import evaluate_case
from shared import cache, config

CASES = load_cases()

# DETERMINISTIC_ONLY=1 runs only the keyless deterministic checks (fast CI tier);
# default runs the full suite including the Claude-judged metrics.
DETERMINISTIC_ONLY = os.getenv("DETERMINISTIC_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(case):
    try:
        report = evaluate_case(case, deterministic_only=DETERMINISTIC_ONLY)
    except cache.CacheMiss:
        pytest.skip(
            f"no recording for {case.id!r} (agent answer or judge verdict) and not in live "
            f"mode — run `make record-missing` once to record the baseline"
        )
    except config.ConfigError as exc:
        pytest.skip(f"missing config for live run: {exc}")

    # Surface judge errors distinctly from real failures.
    judge_errors = [r for r in report.results if r.outcome == Outcome.JUDGE_ERROR]
    assert not judge_errors, f"judge unavailable for {case.id}: {[r.detail for r in judge_errors]}"

    failures = [f"{r.name}: {r.detail}" for r in report.results if r.outcome == Outcome.FAIL]
    assert report.passed, f"{case.id} failed: {failures}"
