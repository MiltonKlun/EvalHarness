"""Metrics-over-time surface (plan 7.2): make drift *visible*, not just pass/fail.

A single pass/fail tells you the state today; it doesn't show a metric slowly sliding
toward the threshold over weeks. This module appends one summary row per live run to a
committed CSV (``evals/history/runs.csv``) and renders a simple text trend, so a gradual
drop in suite-mean faithfulness is visible at a glance — the same way you'd watch a
flaky test's failure rate creep up.

It is intentionally tiny and dependency-free (stdlib ``csv`` only): the history file is
plain CSV so it diffs cleanly in git and a reviewer can read it without tooling.

Two entry points:
  - ``record_run(reports, ...)`` — called by the live runner to append a row.
  - ``python -m evals.history`` — render the committed history as a trend table.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from evals.metrics import CaseReport

HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_CSV = HISTORY_DIR / "runs.csv"

# One row per live run. Kept stable so the CSV stays append-only and git-diffable.
FIELDS = [
    "timestamp",  # UTC ISO-8601, when the run was recorded
    "git_sha",  # commit the run was recorded against (provenance)
    "n_cases",  # cases evaluated
    "pass_rate",  # fraction of cases fully passing
    "mean_faithfulness",  # suite-mean of the faithfulness score (judge metric)
    "mean_relevancy",  # suite-mean of the answer_relevancy score (judge metric)
    "judge_errors",  # count of cases with a JUDGE_ERROR (judge unreachable, not a fail)
]


def _mean_metric(reports: Sequence[CaseReport], metric_name: str) -> float | None:
    """Suite-mean of one judge metric's score across cases that ran it."""
    scores = [
        r.score
        for rep in reports
        for r in rep.results
        if r.name == metric_name and r.score is not None
    ]
    return round(statistics.mean(scores), 4) if scores else None


def summarize(reports: Sequence[CaseReport]) -> dict[str, object]:
    """Reduce a run's per-case reports to the one summary row recorded in history."""
    n = len(reports)
    passed = sum(1 for r in reports if r.passed)
    judge_errors = sum(1 for r in reports if r.had_judge_error)
    return {
        "n_cases": n,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "mean_faithfulness": _mean_metric(reports, "faithfulness"),
        "mean_relevancy": _mean_metric(reports, "answer_relevancy"),
        "judge_errors": judge_errors,
    }


def record_run(
    reports: Sequence[CaseReport],
    *,
    git_sha: str = "",
    timestamp: str | None = None,
    path: Path = HISTORY_CSV,
) -> dict[str, object]:
    """Append one summary row for this run to the history CSV; return the row.

    Creates the file with a header on first write. Append-only by design so history is a
    growing, git-tracked record — never rewritten.
    """
    row = {
        "timestamp": timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        **summarize(reports),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def load_history(path: Path = HISTORY_CSV) -> list[dict[str, str]]:
    """Read all recorded runs (oldest first). Empty list if no history yet."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# The aggregate/regression gate the README's threshold-strategy box promises: a single
# case dipping on one noisy sample isn't a regression, but a suite-wide MEAN drop beyond
# `max_mean_drop` versus the last committed baseline is. These three are "higher is better".
_REGRESSION_METRICS = ("mean_faithfulness", "mean_relevancy", "pass_rate")


def _as_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_regression(current: dict, baseline: dict, max_mean_drop: float) -> list[str]:
    """Return one message per metric that dropped more than ``max_mean_drop`` vs baseline.

    Empty list = pass. A metric that is missing/None on either side is skipped with an
    explanatory message (never silently treated as 0 — that would fabricate a regression).
    ``current`` and ``baseline`` are row dicts (values may be floats or CSV strings).

    Coverage guard: if the current run evaluated FEWER cases than the baseline
    (``n_cases`` shrank), that is itself a regression — the means are computed over a
    smaller, non-comparable set and could be hiding drift on the missing cases. An
    *increase* in n_cases is fine (the dataset grew).
    """
    messages: list[str] = []

    cur_n = _as_float(current.get("n_cases"))
    base_n = _as_float(baseline.get("n_cases"))
    if cur_n is not None and base_n is not None and cur_n < base_n:
        messages.append(
            f"n_cases: COVERAGE SHRANK — {int(base_n)} -> {int(cur_n)} cases; "
            f"the means below are over a smaller, non-comparable set"
        )

    for metric in _REGRESSION_METRICS:
        cur = _as_float(current.get(metric))
        base = _as_float(baseline.get(metric))
        if cur is None or base is None:
            messages.append(f"{metric}: skipped (missing value — cur={cur}, baseline={base})")
            continue
        drop = base - cur
        if drop > max_mean_drop:
            messages.append(
                f"{metric}: REGRESSION — dropped {drop:.4f} "
                f"({base:.4f} -> {cur:.4f}), exceeds max_mean_drop {max_mean_drop}"
            )
    return messages


# Unicode block sparkline (8 levels) with an ASCII fallback for consoles that can't encode
# it (notably the default Windows cp1252 terminal). render() picks based on the stream.
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = ".:-=+*#@"


def _spark(values: Iterable[float | None], blocks: str = _SPARK_BLOCKS) -> str:
    """A tiny inline sparkline so a trend is visible in a plain-text report."""
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        out.append(" " if v is None else blocks[min(7, int((v - lo) / span * 7))])
    return "".join(out)


def _spark_blocks_for(stream: object) -> str:
    """Pick the Unicode sparkline if the target stream can encode it, else ASCII."""
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        _SPARK_BLOCKS.encode(enc)
        return _SPARK_BLOCKS
    except (UnicodeEncodeError, LookupError):
        return _SPARK_ASCII


def render(path: Path = HISTORY_CSV, blocks: str = _SPARK_BLOCKS) -> str:
    """Render the committed history as a trend table + sparklines (for the README/CI).

    ``blocks`` selects the sparkline glyphs; callers writing to a console should pass the
    result of ``_spark_blocks_for(stream)`` so an ASCII-only terminal still renders.
    """
    rows = load_history(path)
    if not rows:
        return "No run history yet. Record one with a live eval run (see evals/history.py)."

    def col(name: str) -> list[float | None]:
        out: list[float | None] = []
        for r in rows:
            raw = r.get(name, "")
            out.append(float(raw) if raw not in ("", None) else None)
        return out

    lines = [
        f"Eval history - {len(rows)} run(s) recorded ({HISTORY_CSV.name})",
        "",
        f"{'timestamp':<20} {'sha':<9} {'pass':>5} {'faith':>6} {'relev':>6} {'jerr':>4}",
        "-" * 56,
    ]
    for r in rows:
        lines.append(
            f"{r['timestamp'][:19]:<20} {(r['git_sha'] or '-')[:8]:<9} "
            f"{r['pass_rate']:>5} {r['mean_faithfulness'] or '-':>6} "
            f"{r['mean_relevancy'] or '-':>6} {r['judge_errors']:>4}"
        )
    lines += [
        "-" * 56,
        f"faithfulness trend: {_spark(col('mean_faithfulness'), blocks)}",
        f"relevancy    trend: {_spark(col('mean_relevancy'), blocks)}",
        f"pass-rate    trend: {_spark(col('pass_rate'), blocks)}",
    ]
    return "\n".join(lines)


def main() -> None:
    import sys

    print(render(blocks=_spark_blocks_for(sys.stdout)))


if __name__ == "__main__":
    main()
