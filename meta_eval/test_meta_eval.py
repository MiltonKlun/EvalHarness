"""Meta-eval gate (plan 6.2/6.4) — asserts the judge agrees with humans well enough.

Replays the cached judge scores (meta_eval/scores.json) and checks that agreement with the
hand-labeled gold set clears a floor. This is also the judge-DRIFT guard: if a future judge
model degrades, re-recording scores and running this turns the suite red.

Keyless (uses cached scores). Skips cleanly if scores haven't been recorded yet.
"""

from __future__ import annotations

import pytest

from meta_eval.run import SCORES_PATH, analyse, collect_scores

# Floors: the judge must agree with the human gold set at least this well to be trusted.
MIN_ACCURACY = 0.80
MIN_KAPPA = 0.60  # "substantial" agreement on the Landis-Koch scale


@pytest.fixture(scope="module")
def analysis():
    if not SCORES_PATH.exists():
        pytest.skip("no recorded judge scores — run `make meta-eval-live` once to record")
    return analyse(collect_scores(live=False))


def test_judge_accuracy_above_floor(analysis):
    assert analysis["accuracy"] >= MIN_ACCURACY, (
        f"judge accuracy {analysis['accuracy']:.1%} < floor {MIN_ACCURACY:.0%} — "
        f"the evaluator may have drifted; see disagreements: {analysis['disagreements']}"
    )


def test_judge_kappa_above_floor(analysis):
    assert analysis["kappa"] >= MIN_KAPPA, (
        f"Cohen's kappa {analysis['kappa']:.2f} < floor {MIN_KAPPA} — judge-human agreement "
        f"is weak; re-examine the judge prompt or model."
    )


def test_calibrated_threshold_is_sane(analysis):
    t = analysis["calibrated_threshold"]
    assert 0.0 <= t <= 1.0
