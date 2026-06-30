"""Offline tests for the metrics-over-time surface — no LLM, no keys."""

from __future__ import annotations

from evals import history
from evals.metrics import CaseReport, MetricResult, Outcome


def _report(
    case_id: str,
    *,
    passed: bool,
    faith: float | None,
    relev: float | None,
    judge_error: bool = False,
) -> CaseReport:
    rep = CaseReport(case_id=case_id)
    det = Outcome.PASS if passed else Outcome.FAIL
    rep.results.append(MetricResult("abstention", det))
    if judge_error:
        rep.results.append(MetricResult("faithfulness", Outcome.JUDGE_ERROR))
    else:
        if faith is not None:
            rep.results.append(
                MetricResult("faithfulness", Outcome.PASS if passed else Outcome.FAIL, score=faith)
            )
        if relev is not None:
            rep.results.append(MetricResult("answer_relevancy", Outcome.PASS, score=relev))
    return rep


def test_summarize_computes_means_and_pass_rate():
    reports = [
        _report("a", passed=True, faith=1.0, relev=0.9),
        _report("b", passed=False, faith=0.0, relev=0.8),
    ]
    s = history.summarize(reports)
    assert s["n_cases"] == 2
    assert s["pass_rate"] == 0.5
    assert s["mean_faithfulness"] == 0.5  # (1.0 + 0.0) / 2
    assert s["mean_relevancy"] == 0.85
    assert s["judge_errors"] == 0


def test_judge_error_counted_not_averaged():
    reports = [
        _report("a", passed=True, faith=1.0, relev=1.0),
        _report("b", passed=False, faith=None, relev=None, judge_error=True),
    ]
    s = history.summarize(reports)
    assert s["judge_errors"] == 1
    assert s["mean_faithfulness"] == 1.0  # the JUDGE_ERROR case contributes no score


def test_record_run_appends_and_roundtrips(tmp_path):
    csv_path = tmp_path / "runs.csv"
    r1 = history.record_run(
        [_report("a", passed=True, faith=0.9, relev=0.9)],
        git_sha="abc1234",
        timestamp="2026-01-01T00:00:00+00:00",
        path=csv_path,
    )
    assert r1["pass_rate"] == 1.0
    history.record_run(
        [_report("a", passed=False, faith=0.4, relev=0.5)],
        git_sha="def5678",
        timestamp="2026-01-08T00:00:00+00:00",
        path=csv_path,
    )
    rows = history.load_history(csv_path)
    assert len(rows) == 2  # append-only: two runs, one header
    assert rows[0]["git_sha"] == "abc1234"
    assert rows[1]["mean_faithfulness"] == "0.4"


def test_render_empty_history_is_graceful(tmp_path):
    out = history.render(tmp_path / "nope.csv")
    assert "No run history yet" in out


def test_render_shows_trend(tmp_path):
    csv_path = tmp_path / "runs.csv"
    history.record_run(
        [_report("a", passed=True, faith=0.9, relev=0.9)],
        timestamp="2026-01-01T00:00:00+00:00",
        path=csv_path,
    )
    history.record_run(
        [_report("a", passed=True, faith=0.7, relev=0.8)],
        timestamp="2026-01-08T00:00:00+00:00",
        path=csv_path,
    )
    out = history.render(csv_path)
    assert "2 run(s) recorded" in out
    assert "faithfulness trend:" in out
