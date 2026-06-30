"""Run the functional suite once and append a summary row to the eval history (plan 7.2).

This is the producer for the metrics-over-time surface: it evaluates every dataset case
(replaying recorded inputs; metrics run live, exactly like ``make eval-ci``) and appends a
single summary row to ``evals/history/runs.csv`` via ``evals.history.record_run``.

Intended for the live tier, where it captures a snapshot per run so drift becomes visible
over time (``python -m evals.history`` renders the trend). It needs an ANTHROPIC key for the
judge metrics; with ``--deterministic-only`` it records pass-rate from the keyless checks alone.

    python -m evals.record_history                 # judged run (needs ANTHROPIC_API_KEY)
    python -m evals.record_history --deterministic-only   # keyless: pass-rate only
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from evals import history
from evals.dataset import load_cases
from evals.runner import evaluate_case
from shared import cache


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return ""


def run(deterministic_only: bool) -> None:
    reports = []
    for case in load_cases():
        try:
            reports.append(evaluate_case(case, deterministic_only=deterministic_only))
        except cache.CacheMiss:
            print(f"  skip {case.id}: no recording yet (run `make eval` to record a baseline)")
    if not reports:
        print("No cases evaluated (no recordings and no live keys) — nothing recorded.")
        return
    row = history.record_run(reports, git_sha=_git_sha())
    print(f"Recorded run: {row}")
    print()
    print(history.render(blocks=history._spark_blocks_for(sys.stdout)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one eval run into evals/history/.")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="skip the Claude judge; record pass-rate from the keyless checks only",
    )
    run(parser.parse_args().deterministic_only)


if __name__ == "__main__":
    main()
