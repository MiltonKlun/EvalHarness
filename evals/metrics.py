"""Metric assertions for the functional eval suite (plan 2.2).

Two layers, cheapest first:

1. Deterministic, NON-LLM checks (free, instant, no judge): citation present, expected
   sources cited, abstention on unanswerable cases, must_contain substrings. These are
   the first line of defence — if they fail, we don't even pay for a judge call.

2. LLM-as-judge metrics (Claude): faithfulness (grounded in context) and answer
   relevancy. These run only on answerable/multihop cases.

Judge-failure handling (operational maturity): a judge call that errors/times out/rate-
limits is retried with backoff, then surfaced as a distinct JUDGE_ERROR outcome — never
a silent pass. A run containing any JUDGE_ERROR is flagged, not green.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from evals.dataset import Case

ABSTENTION_STRING = "i don't know based on the provided documents"


class Outcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    JUDGE_ERROR = "JUDGE_ERROR"  # judge unavailable — distinct from a real fail


@dataclass
class MetricResult:
    name: str
    outcome: Outcome
    score: float | None = None  # 0..1 for LLM metrics; None for deterministic checks
    detail: str = ""


@dataclass
class CaseReport:
    case_id: str
    results: list[MetricResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.outcome == Outcome.PASS for r in self.results)

    @property
    def had_judge_error(self) -> bool:
        return any(r.outcome == Outcome.JUDGE_ERROR for r in self.results)


# --- Layer 1: deterministic checks (no LLM, no cost) -------------------------------


def _is_abstention(answer: str) -> bool:
    return ABSTENTION_STRING in answer.lower()


def check_abstention(case: Case, answer: str) -> MetricResult:
    """Unanswerable cases must abstain; answerable cases must NOT abstain."""
    abstained = _is_abstention(answer)
    if case.is_unanswerable:
        ok = abstained
        detail = "abstained as required" if ok else "INVENTED an answer instead of abstaining"
    else:
        ok = not abstained
        detail = "answered" if ok else "wrongly abstained on an answerable question"
    return MetricResult("abstention", Outcome.PASS if ok else Outcome.FAIL, detail=detail)


def check_must_contain(case: Case, answer: str) -> MetricResult:
    """Answer must contain the key fact substrings (case-insensitive)."""
    if not case.must_contain:
        return MetricResult("must_contain", Outcome.PASS, detail="n/a")
    low = answer.lower()
    missing = [s for s in case.must_contain if s.lower() not in low]
    ok = not missing
    return MetricResult(
        "must_contain",
        Outcome.PASS if ok else Outcome.FAIL,
        detail="all present" if ok else f"missing: {missing}",
    )


def check_sources(case: Case, sources_cited: list[str]) -> MetricResult:
    """Answerable cases must cite all expected sources (verifiable from contexts)."""
    if case.is_unanswerable or not case.expected_sources:
        return MetricResult("sources", Outcome.PASS, detail="n/a")
    cited = set(sources_cited)
    missing = [s for s in case.expected_sources if s not in cited]
    ok = not missing
    return MetricResult(
        "sources",
        Outcome.PASS if ok else Outcome.FAIL,
        detail="expected sources present" if ok else f"missing sources: {missing}",
    )


# --- Layer 2: LLM-as-judge metrics (Claude), with judge-failure handling -----------


def _find_in_chain(exc: BaseException, types: tuple[type, ...]) -> BaseException | None:
    """Walk an exception's __cause__/__context__ chain for an instance of ``types``.

    DeepEval catches errors raised inside the judge and re-raises its own wrapper, so a
    bare ``except CacheMiss`` on ``metric.measure`` can miss it. We inspect the chain so a
    propagated CacheMiss / ConfigError is recognised even when wrapped.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, types):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


def _run_judge_metric(metric, test_case, name: str, threshold: float) -> MetricResult:
    """Run a DeepEval metric with retry/backoff; map failures to JUDGE_ERROR.

    A *low score* is a real FAIL. A judge *exception* (network/timeout/rate-limit) is a
    JUDGE_ERROR — we never let an unreachable judge look like a pass.

    Two failures are NOT transient and must propagate immediately, not retry-sleep into a
    JUDGE_ERROR: a ``CacheMiss`` (replay mode, this verdict not recorded — the runner turns
    it into a SKIP) and a ``ConfigError`` (no key — also a SKIP upstream). Retrying either
    would waste seconds of backoff on a certainty. Both are detected even when DeepEval
    wraps them (see ``_find_in_chain``).
    """
    from shared.cache import CacheMiss
    from shared.config import ConfigError

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            metric.measure(test_case)
            score = metric.score
            ok = score is not None and score >= threshold
            return MetricResult(
                name,
                Outcome.PASS if ok else Outcome.FAIL,
                score=score,
                detail=getattr(metric, "reason", "") or "",
            )
        except Exception as exc:  # noqa: BLE001 - classify, then either propagate or retry
            propagate = _find_in_chain(exc, (CacheMiss, ConfigError))
            if propagate is not None:
                # Re-surface the root cause (CacheMiss/ConfigError), dropping DeepEval's
                # wrapper; not transient — no retries, no JUDGE_ERROR.
                raise propagate from None
            last_exc = exc
            time.sleep(2**attempt)  # 1s, 2s, 4s backoff
    return MetricResult(
        name,
        Outcome.JUDGE_ERROR,
        detail=f"judge unavailable after retries: {type(last_exc).__name__}: {last_exc}",
    )


def llm_metrics(case: Case, answer: str, contexts: list[str], judge, thresholds: dict):
    """Faithfulness + relevancy via Claude. Skipped for unanswerable cases."""
    if case.is_unanswerable:
        return []

    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=case.question,
        actual_output=answer,
        retrieval_context=contexts,
    )
    per = thresholds["per_answer"]
    return [
        _run_judge_metric(
            FaithfulnessMetric(model=judge, threshold=per["faithfulness"]),
            test_case,
            "faithfulness",
            per["faithfulness"],
        ),
        _run_judge_metric(
            AnswerRelevancyMetric(model=judge, threshold=per["answer_relevancy"]),
            test_case,
            "answer_relevancy",
            per["answer_relevancy"],
        ),
    ]
