"""Run the functional suite once and append a summary row to the eval history (plan 7.2).

This is the producer for the metrics-over-time surface: it evaluates every dataset case
(replaying recorded inputs; metrics run live, exactly like ``make eval-ci``) and appends a
single summary row to ``evals/history/runs.csv`` via ``evals.history.record_run``.

Intended for the live tier, where it captures a snapshot per run so drift becomes visible
over time (``python -m evals.history`` renders the trend). It needs an ANTHROPIC key for the
judge metrics; with ``--deterministic-only`` it records pass-rate from the keyless checks alone.

After appending the row it runs the **aggregate regression gate**: the new suite means are
compared against the last committed row, and a drop beyond ``regression.max_mean_drop``
(``thresholds.yaml``) exits non-zero — this is the drift alarm the README promises, and it
is what makes a scheduled live run go red on genuine degradation. ``--no-gate`` bypasses it
(first run / intentional baseline change).

    python -m evals.record_history                 # judged run (needs ANTHROPIC_API_KEY)
    python -m evals.record_history --deterministic-only   # keyless: pass-rate only
    python -m evals.record_history --no-gate       # record without the regression gate
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from evals import history
from evals.dataset import load_cases, load_thresholds
from evals.runner import evaluate_case
from shared import cache


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return ""


def run(deterministic_only: bool, gate: bool = True) -> None:
    # Capture the current last row BEFORE appending — that's the regression baseline.
    prior = history.load_history()
    baseline = prior[-1] if prior else None

    reports = []
    for case in load_cases():
        try:
            reports.append(evaluate_case(case, deterministic_only=deterministic_only))
        except cache.CacheMiss:
            print(f"  skip {case.id}: no recording yet (run `make record-missing` to record)")
    if not reports:
        print("No cases evaluated (no recordings and no live keys) — nothing recorded.")
        return

    row = history.record_run(reports, git_sha=_git_sha())
    print(f"Recorded run: {row}")
    # Machine-readable signal for CI: did the METRICS (not the timestamp/sha) change vs the
    # last row? The auto-commit step commits the row only when this is yes — a flat trend
    # must not spawn a weekly noise commit.
    changed = history.metrics_changed(row, baseline)
    print(f"METRICS_CHANGED: {'yes' if changed else 'no'}")
    print()
    print(history.render(blocks=history._spark_blocks_for(sys.stdout)))

    if not gate:
        print("\nRegression gate: SKIPPED (--no-gate).")
        return
    if baseline is None:
        print("\nRegression gate: no prior baseline row — first run, nothing to compare.")
        return

    max_drop = load_thresholds()["regression"]["max_mean_drop"]
    failures = history.check_regression(row, baseline, max_drop)
    print(f"\nRegression gate (vs {baseline.get('git_sha', '?')}, max_mean_drop={max_drop}):")
    if failures:
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("  PASS — no suite-mean dropped beyond the threshold.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one eval run into evals/history/.")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="skip the Claude judge; record pass-rate from the keyless checks only",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="record the row but skip the regression gate (first run / intentional baseline)",
    )
    args = parser.parse_args()
    run(args.deterministic_only, gate=not args.no_gate)


if __name__ == "__main__":
    main()
