"""Run the functional eval suite over the dataset (plan 2.3/2.4).

For each case:
  1. get the RAG answer via app.rag.answer() — which routes the Gemini call through the
     record/replay cache (LIVE_LLM=1 records; default replays). This is the 2.3 wiring:
     inputs are replayed offline, but the metrics below always run live.
  2. run deterministic checks (abstention, must_contain, sources) — free, no judge.
  3. run the Claude judge metrics (faithfulness, relevancy) on answerable cases.

Used by both the pytest runner (test_functional.py) and `make eval` / `make eval-ci`.
"""

from __future__ import annotations

from evals import metrics
from evals.dataset import Case, load_thresholds


def evaluate_case(case: Case, judge=None) -> metrics.CaseReport:
    """Evaluate one case end-to-end and return its report.

    ``judge`` is injected so tests can pass a fake; in real runs it defaults to the
    Claude judge (constructed lazily, only if an LLM metric actually runs).
    """
    from app.rag import answer as rag_answer

    result = rag_answer(case.question)  # {answer, contexts, sources}
    ans, contexts, sources = result["answer"], result["contexts"], result["sources"]

    report = metrics.CaseReport(case_id=case.id)
    # Layer 1: deterministic (always).
    report.results.append(metrics.check_abstention(case, ans))
    report.results.append(metrics.check_must_contain(case, ans))
    report.results.append(metrics.check_sources(case, sources))

    # Layer 2: judge (answerable/multihop only).
    if not case.is_unanswerable:
        if judge is None:
            from evals.judge import ClaudeJudge

            judge = ClaudeJudge()
        report.results.extend(metrics.llm_metrics(case, ans, contexts, judge, load_thresholds()))
    return report
