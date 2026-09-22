"""Experiment 4 (control) -- NOT YET RUN: blocked on Anthropic credit (2026-09-22).

Run:  JUDGE_LIVE=1 .venv/Scripts/python.exe docs/jev-evidence/run_exp4_idk_control.py

Experiment 4 (control): is JUDGE-001 Claude's judgment, or DeepEval's arithmetic?

DeepEval's verdict prompt tells the judge: "Use 'idk' for claims not backed up by context".
Its default scorer then counts 'idk' as FAITHFUL (only 'no' is penalised);
penalize_ambiguous_claims=True counts 'idk' as unfaithful instead.

Run with JUDGE_LIVE=1: the Claude judge is called fresh and the cache is bypassed entirely
(no read, no write), so the committed baseline cannot be touched. Each case is judged ONCE;
both scores are computed from that SAME set of verdicts -- identical judgments, two rules.
"""

import json
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
SCRATCH = str(Path(__file__).resolve().parent)
sys.path.insert(0, REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO + "/.env")

from deepeval.metrics import FaithfulnessMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.judge import ClaudeJudge  # noqa: E402
from meta_eval import stats  # noqa: E402
from meta_eval.run import SCORES_PATH, load_gold  # noqa: E402
from shared import config  # noqa: E402

assert config.JUDGE_LIVE, "must run with JUDGE_LIVE=1 so the committed cache is untouched"

KNOWN_FP = {
    "g_correct_world_fact_unsupported",
    "g_overconfident_inference",
    "g_invented_specific",
    "g_hedge_then_invent",
}


def score(verdicts: list[str], penalize_idk: bool) -> float:
    """DeepEval's _calculate_score, reproduced exactly (faithfulness.py:375)."""
    if not verdicts:
        return 1.0
    n = sum(1 for v in verdicts if v != "no")
    if penalize_idk:
        n -= sum(1 for v in verdicts if v == "idk")
    return n / len(verdicts)


gold = load_gold()
committed = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
judge = ClaudeJudge()
rows = []

for case in gold:
    tc = LLMTestCase(
        input="Is the answer faithful to the context?",
        actual_output=case["answer"],
        retrieval_context=case["context"],
    )
    m = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=False)
    m.measure(tc)
    verdicts = [v.verdict.strip().lower() for v in m.verdicts]
    rows.append(
        {
            "id": case["id"],
            "human": case["human_verdict"],
            "committed": committed[case["id"]],
            "deepeval_score": round(float(m.score), 3),
            "default": round(score(verdicts, False), 3),
            "penalize_idk": round(score(verdicts, True), 3),
            "claims": list(m.claims),
            "verdicts": verdicts,
            "reasons": [v.reason for v in m.verdicts],
        }
    )
    r = rows[-1]
    flag = "  <-- JUDGE-001 FP" if case["id"] in KNOWN_FP else ""
    print(
        f"{r['id']:38} {r['human']:10} commit={r['committed']:.2f} "
        f"default={r['default']:.2f} pen_idk={r['penalize_idk']:.2f}  {verdicts}{flag}",
        flush=True,
    )

# Sanity: my reproduction of the scoring rule must equal DeepEval's own score.
bad = [r["id"] for r in rows if abs(r["default"] - r["deepeval_score"]) > 1e-6]
print(f"\nscoring-rule reproduction mismatches: {len(bad)} {bad}")

human = [r["human"] for r in rows]
for label in ("committed", "default", "penalize_idk"):
    s = [r[label] for r in rows]
    t, _ = stats.calibrate_threshold(human, s)
    pred = [stats.verdict_from_score(x, t) for x in s]
    c = stats.confusion(human, pred)
    k = stats.cohen_kappa(human, pred)
    print(
        f"[{label:12}] best cutoff {t:.2f}: accuracy {c.accuracy:.0%}  kappa {k:.2f}  "
        f"tp={c.tp} tn={c.tn} fp={c.fp} fn={c.fn}"
    )

with open(SCRATCH + "/exp4_idk_control.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
print("\nwrote exp4_raw.json")
