"""Offline tests for the metric layer — no API keys, no network.

These verify the deterministic checks and the judge-failure handling logic directly,
with stub answers and a fake judge. The full LLM-judged suite (test_functional.py)
needs recordings/keys and is gated separately.
"""

from __future__ import annotations

from evals import metrics
from evals.dataset import Case
from evals.metrics import Outcome

ANSWERABLE = Case(
    id="t_ans",
    type="answerable",
    question="Who founded Meridian?",
    reference_answer="Solvang and Vale.",
    expected_sources=["01_company_overview.md"],
    must_contain=["Solvang", "Vale"],
)
UNANSWERABLE = Case(
    id="t_unans",
    type="unanswerable",
    question="What is the revenue?",
    reference_answer="Not disclosed.",
    expected_sources=[],
    abstain=True,
)


def test_abstention_required_on_unanswerable():
    bad = metrics.check_abstention(UNANSWERABLE, "The revenue was $5M.")  # invented
    good = metrics.check_abstention(UNANSWERABLE, "I don't know based on the provided documents.")
    assert bad.outcome == Outcome.FAIL
    assert good.outcome == Outcome.PASS


def test_abstention_not_allowed_on_answerable():
    wrong = metrics.check_abstention(ANSWERABLE, "I don't know based on the provided documents.")
    assert wrong.outcome == Outcome.FAIL  # answerable q must not abstain


def test_must_contain():
    ok = metrics.check_must_contain(ANSWERABLE, "Founded by Ada Solvang and Henrik Vale.")
    bad = metrics.check_must_contain(ANSWERABLE, "Founded by two people.")
    assert ok.outcome == Outcome.PASS
    assert bad.outcome == Outcome.FAIL


def test_sources_must_be_cited():
    ok = metrics.check_sources(ANSWERABLE, ["01_company_overview.md"])
    bad = metrics.check_sources(ANSWERABLE, ["02_products.md"])
    assert ok.outcome == Outcome.PASS
    assert bad.outcome == Outcome.FAIL


def test_judge_exception_becomes_judge_error_not_pass():
    """A judge that always raises must produce JUDGE_ERROR, never a silent PASS."""

    class _Boom:
        score = None
        reason = ""

        def measure(self, _tc):
            raise RuntimeError("simulated rate limit")

    res = metrics._run_judge_metric(_Boom(), object(), "faithfulness", 0.7)
    assert res.outcome == Outcome.JUDGE_ERROR


def test_low_score_is_real_fail_not_judge_error():
    class _LowScore:
        score = 0.2
        reason = "ungrounded"

        def measure(self, _tc):
            return None

    res = metrics._run_judge_metric(_LowScore(), object(), "faithfulness", 0.7)
    assert res.outcome == Outcome.FAIL
    assert res.score == 0.2
