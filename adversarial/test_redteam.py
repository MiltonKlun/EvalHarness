"""Parametrized red-team suite — one pytest case per attack payload.

Behaviour mirrors the functional suite:
  - LIVE_LLM=1  -> real agent calls, recorded to evals/cache (make redteam-live / live tier).
  - default     -> replay recorded agent responses; grading runs live. If a case has no
                   recording, it SKIPS (so the plain offline run stays green pre-baseline).

A case FAILS on `breach`. `partial_leak` is allowed-but-reported (a hairline outcome we
deliberately surface rather than hide). `judge_error` fails loudly (an unreachable toxicity
judge is never silently 'safe').

DETERMINISTIC_ONLY=1 skips the Claude-judged toxicity cases (keyless fast-CI tier).
"""

from __future__ import annotations

import os

import pytest

from adversarial.grade import Grade
from adversarial.run import load_payloads, run_case
from shared import config

DETERMINISTIC_ONLY = os.getenv("DETERMINISTIC_ONLY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Flatten (category, judged, case) tuples for parametrization.
_CASES = [
    (p["category"], p.get("judged", False), case) for p in load_payloads() for case in p["cases"]
]


@pytest.mark.parametrize(
    ("category", "judged", "case"),
    _CASES,
    ids=[f"{p['category']}-{c['id']}" for p in load_payloads() for c in p["cases"]],
)
def test_attack(category, judged, case):
    if judged and DETERMINISTIC_ONLY:
        # The toxicity verdict is now cached and replays offline, so this CAN run keyless.
        # We still keep it out of the blocking/deterministic tier on purpose: judge-graded
        # checks are deliberately non-blocking (see the README CI-philosophy section). It
        # runs in the judged + live tiers, not on every PR.
        pytest.skip("judge-graded toxicity case is deliberately non-blocking (judged/live tiers)")

    try:
        result = run_case(category, judged, case)
    except config.ConfigError as exc:
        pytest.skip(f"missing config: {exc}")

    # run_case surfaces an unrecorded case as NOT_RECORDED (so the standalone `make redteam`
    # runner degrades gracefully); in pytest that's not real coverage, so skip it.
    if result.grade == Grade.NOT_RECORDED:
        pytest.skip(f"no recording for {case['id']!r} — run `make redteam-live` to record")

    assert result.grade != Grade.JUDGE_ERROR, f"toxicity judge unavailable: {result.detail}"
    assert result.grade != Grade.BREACH, (
        f"BREACH on {case['id']}: {result.detail}\nanswer: {result.answer[:200]}"
    )
    # partial_leak is intentionally allowed but visible — print it for the report.
    if result.grade == Grade.PARTIAL_LEAK:
        print(f"PARTIAL_LEAK {case['id']}: {result.detail}")
