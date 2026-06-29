"""Offline tests for the meta-eval statistics — no LLM, no keys."""

from __future__ import annotations

from meta_eval import stats
from meta_eval.stats import GROUNDED as G
from meta_eval.stats import UNGROUNDED as U


def test_confusion_counts():
    human = [G, G, U, U]
    judge = [G, U, U, G]  # one FN (2nd), one FP (4th)
    c = stats.confusion(human, judge)
    assert (c.tp, c.fn, c.tn, c.fp) == (1, 1, 1, 1)
    assert c.accuracy == 0.5


def test_perfect_agreement_kappa_is_one():
    human = [G, U, G, U]
    assert stats.cohen_kappa(human, human) == 1.0


def test_chance_agreement_kappa_near_zero():
    # Judge ignores truth and always says grounded; humans split 50/50.
    human = [G, U, G, U]
    judge = [G, G, G, G]
    k = stats.cohen_kappa(human, judge)
    assert abs(k) < 1e-9  # no agreement beyond chance


def test_verdict_from_score():
    assert stats.verdict_from_score(0.8, 0.7) == G
    assert stats.verdict_from_score(0.6, 0.7) == U


def test_calibrate_picks_separating_threshold():
    # grounded cases score high, ungrounded score low -> a threshold ~0.5 separates them.
    human = [G, G, U, U]
    scores = [0.9, 0.8, 0.2, 0.1]
    t, acc = stats.calibrate_threshold(human, scores)
    assert acc == 1.0
    assert 0.2 < t <= 0.8


def test_calibrate_handles_imperfect_separation():
    human = [G, G, U, U]
    scores = [0.9, 0.4, 0.6, 0.1]  # overlap: one grounded low, one ungrounded high
    _, acc = stats.calibrate_threshold(human, scores)
    assert acc < 1.0  # can't perfectly separate
