"""Meta-eval runner (plan 6.2-6.4): measure the judge against a human gold set.

Flow:
  1. For each gold case, get the Claude judge's faithfulness SCORE (0..1) for (answer,
     context). This is the only step that calls the LLM; scores are cached to
     meta_eval/scores.json so calibration/analysis re-run offline.
  2. Calibrate the threshold that best separates the human grounded/ungrounded labels.
  3. Report agreement at that threshold: accuracy, confusion matrix, Cohen's kappa, and
     the specific cases where the judge disagrees with the human (its failure modes).
  4. Optionally write the calibrated threshold into thresholds.yaml.

Re-running this is also the JUDGE-DRIFT check (6.4): if agreement/kappa drops over time,
the judge model has drifted and thresholds may need re-calibrating.

    python -m meta_eval.run            # measure + report
    python -m meta_eval.run --write    # also write calibrated threshold to thresholds.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meta_eval import stats

_DIR = Path(__file__).resolve().parent
GOLD_PATH = _DIR / "gold.jsonl"
SCORES_PATH = _DIR / "scores.json"  # cached judge scores (committed for offline re-runs)
_REPO_ROOT = _DIR.parent
THRESHOLDS_PATH = _REPO_ROOT / "thresholds.yaml"

# A safety margin subtracted from the raw calibrated cutoff so we don't sit a hair above
# the judge's decision boundary (the judge is itself noisy — see the Phase 2 flake finding).
THRESHOLD_MARGIN = 0.05


def load_gold() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _judge_score(case: dict, judge) -> float:
    """Faithfulness score (0..1) for one gold case from the Claude judge."""
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    metric = FaithfulnessMetric(model=judge, threshold=0.5)  # threshold unused; we read .score
    metric.measure(
        LLMTestCase(
            input="Is the answer faithful to the context?",
            actual_output=case["answer"],
            retrieval_context=case["context"],
        )
    )
    return float(metric.score)


def collect_scores(live: bool) -> dict[str, float]:
    """Return {case_id: judge_score}. Live = call the judge + cache; else read cache."""
    if not live and SCORES_PATH.exists():
        return json.loads(SCORES_PATH.read_text(encoding="utf-8"))

    from evals.judge import ClaudeJudge

    judge = ClaudeJudge()
    scores = {c["id"]: _judge_score(c, judge) for c in load_gold()}
    SCORES_PATH.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def analyse(scores: dict[str, float]) -> dict:
    """Calibrate + compute agreement stats + collect disagreements."""
    gold = load_gold()
    human = [c["human_verdict"] for c in gold]
    score_list = [scores[c["id"]] for c in gold]

    best_t, best_acc = stats.calibrate_threshold(human, score_list)
    calibrated = max(0.0, round(best_t - THRESHOLD_MARGIN, 2))

    predicted = [stats.verdict_from_score(scores[c["id"]], best_t) for c in gold]
    conf = stats.confusion(human, predicted)
    kappa = stats.cohen_kappa(human, predicted)

    disagreements = [
        {
            "id": c["id"],
            "human": c["human_verdict"],
            "judge": p,
            "score": scores[c["id"]],
            "note": c.get("note", ""),
        }
        for c, p in zip(gold, predicted, strict=True)
        if c["human_verdict"] != p
    ]
    return {
        "n": len(gold),
        "best_threshold": best_t,
        "calibrated_threshold": calibrated,
        "accuracy": conf.accuracy,
        "kappa": kappa,
        "confusion": conf,
        "disagreements": disagreements,
    }


def format_report(a: dict) -> str:
    c = a["confusion"]
    lines = [
        "# Meta-eval: how good is the Claude judge?\n",
        f"gold cases:             {a['n']}",
        f"best threshold:         {a['best_threshold']:.2f}  (max agreement with humans)",
        f"calibrated threshold:   {a['calibrated_threshold']:.2f}  (best - {THRESHOLD_MARGIN})",
        f"accuracy @ best:        {a['accuracy']:.1%}",
        f"Cohen's kappa:          {a['kappa']:.2f}",
        "",
        "confusion (positive = grounded):",
        f"  TP={c.tp}  FN={c.fn}  (judge too strict -> flagged a good answer)",
        f"  FP={c.fp}  TN={c.tn}  (judge too lenient -> missed a hallucination)",
        "",
        f"disagreements ({len(a['disagreements'])}):",
    ]
    for d in a["disagreements"]:
        lines.append(
            f"  {d['id']:28} human={d['human']:10} judge={d['judge']:10} "
            f"score={d['score']:.2f}  {d['note'][:50]}"
        )
    if not a["disagreements"]:
        lines.append("  (none — perfect agreement on this gold set)")
    return "\n".join(lines)


def write_threshold(a: dict, run_id: str) -> None:
    """Write the calibrated faithfulness threshold into thresholds.yaml."""
    import yaml

    data = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    data["calibrated"] = True
    data["calibration_run"] = run_id
    data["per_answer"]["faithfulness"] = a["calibrated_threshold"]
    header = (
        "# Metric pass/fail thresholds — CALIBRATED by the Phase 6 meta-eval.\n"
        f"# faithfulness was set by `make meta-eval` (run {run_id}): the cutoff that best\n"
        "# separates the hand-labeled gold set, minus a 0.05 margin for the judge's noise.\n"
        "# The judge has a documented lenient bias (JUDGE-001 in adversarial/FINDINGS.md) —\n"
        "# ~80% accuracy, so these scores are an UPPER BOUND on true groundedness.\n"
        "# Re-running the meta-eval (judge-drift check) re-derives these values.\n"
    )
    THRESHOLDS_PATH.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(
        f"\nWrote calibrated faithfulness threshold {a['calibrated_threshold']} to thresholds.yaml"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-eval: challenge the judge.")
    parser.add_argument(
        "--live", action="store_true", help="call the judge (else use cached scores)"
    )
    parser.add_argument(
        "--write", action="store_true", help="write calibrated threshold to thresholds.yaml"
    )
    args = parser.parse_args()

    scores = collect_scores(live=args.live)
    analysis = analyse(scores)
    print(format_report(analysis))
    if args.write:
        import datetime

        write_threshold(analysis, run_id=f"meta-eval-{datetime.date.today().isoformat()}")


if __name__ == "__main__":
    main()
