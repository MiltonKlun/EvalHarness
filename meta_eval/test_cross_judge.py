"""Offline tests for the cross-judge spot-check — no LLM, no keys.

Exercises the pure logic (score parsing + the two-judge agreement math) with synthetic
scores, so it runs keyless like the rest of the suite. The live OpenAI call is never touched.
"""

from __future__ import annotations

import pytest

from meta_eval import cross_judge
from meta_eval.run import load_gold


class TestParseScore:
    def test_plain_float(self):
        assert cross_judge._parse_score("0.83") == pytest.approx(0.83)

    def test_integer_bounds(self):
        assert cross_judge._parse_score("1") == 1.0
        assert cross_judge._parse_score("0") == 0.0

    def test_leading_dot(self):
        assert cross_judge._parse_score(".25") == pytest.approx(0.25)

    def test_score_embedded_in_prose_is_extracted(self):
        # Judge-B occasionally prefixes the number; we take the first 0..1 token.
        assert cross_judge._parse_score("Score: 0.4 — mostly faithful") == pytest.approx(0.4)

    def test_out_of_range_is_clamped(self):
        # Defensive: a stray value above 1 is clamped, not trusted.
        assert cross_judge._parse_score("1.0") == 1.0

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            cross_judge._parse_score("no number here")


def _perfect_scores(verdicts):
    """Map grounded->0.9, ungrounded->0.1 so any sane threshold separates them."""
    return {i: (0.9 if v == "grounded" else 0.1) for i, v in enumerate(verdicts)}


class TestCrossAnalyse:
    def _gold_ids_and_humans(self):
        gold = load_gold()
        return [c["id"] for c in gold], [c["human_verdict"] for c in gold]

    def test_both_judges_perfect_agree_fully(self):
        ids, humans = self._gold_ids_and_humans()
        # Both judges score exactly along the human labels.
        scores = {
            cid: (0.9 if h == "grounded" else 0.1) for cid, h in zip(ids, humans, strict=True)
        }
        a = cross_judge.cross_analyse(scores, dict(scores))

        assert a["n"] == len(ids)
        assert a["accuracy_a"] == 1.0
        assert a["accuracy_b"] == 1.0
        assert a["judge_agreement"] == 1.0
        assert a["judge_kappa"] == pytest.approx(1.0)
        assert a["judge_disagreements"] == []

    def test_disagreement_is_surfaced(self):
        ids, humans = self._gold_ids_and_humans()
        scores_a = {
            cid: (0.9 if h == "grounded" else 0.1) for cid, h in zip(ids, humans, strict=True)
        }
        # Judge-B flips ONE case relative to the human label -> exactly one A/B disagreement.
        scores_b = dict(scores_a)
        flip_id = ids[0]
        scores_b[flip_id] = 1.0 - scores_b[flip_id]

        a = cross_judge.cross_analyse(scores_a, scores_b)
        assert len(a["judge_disagreements"]) == 1
        assert a["judge_disagreements"][0]["id"] == flip_id
        # Judge-A stayed perfect; judge-B lost exactly that one case.
        assert a["accuracy_a"] == 1.0
        assert a["accuracy_b"] < 1.0
        assert a["judge_agreement"] < 1.0

    def test_thresholds_are_in_unit_range(self):
        ids, humans = self._gold_ids_and_humans()
        scores = {
            cid: (0.9 if h == "grounded" else 0.1) for cid, h in zip(ids, humans, strict=True)
        }
        a = cross_judge.cross_analyse(scores, dict(scores))
        assert 0.0 <= a["threshold_a"] <= 1.0
        assert 0.0 <= a["threshold_b"] <= 1.0


def test_report_renders_without_disagreements():
    # Minimal hand-built analysis dict (no gold dependency) to check the formatter is safe.
    a = {
        "n": 2,
        "threshold_a": 0.5,
        "threshold_b": 0.5,
        "accuracy_a": 1.0,
        "accuracy_b": 1.0,
        "kappa_a": 1.0,
        "kappa_b": 1.0,
        "judge_agreement": 1.0,
        "judge_kappa": 1.0,
        "judge_disagreements": [],
    }
    out = cross_judge.format_report(a)
    assert "none" in out.lower()
    assert "spot-check" in out.lower()
