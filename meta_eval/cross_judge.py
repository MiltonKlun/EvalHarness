"""Cross-judge spot-check (build plan 6.5, deferred) — a SECOND judge, for the write-up.

The pinned judge (judge-A) is Claude Haiku; the system under test is Gemini. This adds a
THIRD family — OpenAI — purely as an illustrative spot-check: score the same gold set with
judge-B, then report where the two judges (dis)agree and how each sits against the human
labels. It answers "would a different family's judge reach the same verdicts?" for the
meta-eval story.

**This is a spot-check, not a gate** (locked decision): it never blocks CI, never touches the
pinned ``JUDGE_MODEL`` or the calibrated thresholds, and never enters the metric path. The
committed judge is unchanged; this only reads.

Record/replay: judge-B's raw scores are routed through ``shared.cache`` exactly like every
other model call. Recording needs OPENAI_API_KEY live once (``LIVE_LLM=1``); after that the
scores replay from ``meta_eval/scores_openai.json`` with no key and no network, so a keyless
clone runs this offline like the rest of the suite.

    python -m meta_eval.cross_judge            # replay cached judge-B scores + report
    LIVE_LLM=1 python -m meta_eval.cross_judge  # record judge-B scores once, then report
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from meta_eval import stats
from meta_eval.run import GOLD_PATH, SCORES_PATH, collect_scores, load_gold
from shared import cache, config

_DIR = Path(__file__).resolve().parent
# Judge-B cached scores — committed, so the spot-check replays offline like everything else.
SCORES_B_PATH = _DIR / "scores_openai.json"

# Judge-B: a third family (OpenAI), deliberately distinct from Gemini (generator) and Claude
# (judge-A). Pinned like every other model; overridable for a different spot-check.
JUDGE_B_MODEL = "openai:gpt-4o"

# Faithfulness prompt for judge-B. Kept deliberately close in spirit to what DeepEval's
# FaithfulnessMetric asks judge-A, so the two judges are answering the same question: does
# EVERY claim in the answer follow from the context? We ask for a bare 0..1 score to keep the
# parse trivial and the cache key stable.
_PROMPT_TEMPLATE = """You are grading whether an answer is FAITHFUL to its context.

An answer is faithful only if every factual claim in it is supported by the context.
- Contradicting the context, inventing specifics not in the context, or adding
  world-true-but-unstated facts all make it UNFAITHFUL.
- A faithful subset (omitting some context) or a correct paraphrase is still faithful.
- Abstaining ("I don't know from the documents") when the context withholds the answer is
  faithful.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a single number between 0 and 1 (inclusive) — the fraction of the answer's
claims that are supported by the context. 1 = fully faithful, 0 = wholly unsupported."""


def _parse_score(text: str) -> float:
    """Pull a 0..1 float out of judge-B's reply; clamp to [0, 1]."""
    m = re.search(r"[01](?:\.\d+)?|\.\d+", text.strip())
    if not m:
        raise ValueError(f"could not parse a 0..1 score from judge-B reply: {text!r}")
    return max(0.0, min(1.0, float(m.group())))


def _judge_b_score(case: dict) -> float:
    """Faithfulness score (0..1) for one gold case from judge-B, via record/replay.

    In replay mode this returns the recorded score with no OpenAI key and no network. In live
    mode it makes one real OpenAI call and records it. The key check happens only inside the
    compute closure, so replay never needs a key (same discipline as ``evals/judge.py``).
    """
    context = "\n".join(case["context"])
    prompt = _PROMPT_TEMPLATE.format(context=context, answer=case["answer"])
    params = {"kind": "faithfulness_score", "max_tokens": 16}

    def _compute() -> str:
        from openai import OpenAI

        config.require("OPENAI_API_KEY")
        client = OpenAI()
        model_id = JUDGE_B_MODEL.split(":", 1)[-1]
        resp = client.chat.completions.create(
            model=model_id,
            max_tokens=params["max_tokens"],
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    raw = cache.cached_call(JUDGE_B_MODEL, prompt, params, _compute)
    return _parse_score(raw)


def collect_scores_b(live: bool) -> dict[str, float]:
    """Return {case_id: judge_B_score}. Replay reads the committed cache; live records it."""
    if not live and SCORES_B_PATH.exists():
        return json.loads(SCORES_B_PATH.read_text(encoding="utf-8"))

    scores = {c["id"]: _judge_b_score(c) for c in load_gold()}
    SCORES_B_PATH.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def cross_analyse(scores_a: dict[str, float], scores_b: dict[str, float]) -> dict:
    """Compare judge-A and judge-B on the gold set, each against the humans and each other.

    Each judge is calibrated on its OWN scores (its best-separating threshold), so we compare
    the two judges at their fair operating points rather than forcing judge-B onto judge-A's
    cutoff. Returns per-judge accuracy/kappa vs humans, judge-vs-judge agreement, and the
    specific cases where the two judges disagree.
    """
    gold = load_gold()
    human = [c["human_verdict"] for c in gold]

    a_list = [scores_a[c["id"]] for c in gold]
    b_list = [scores_b[c["id"]] for c in gold]

    t_a, _ = stats.calibrate_threshold(human, a_list)
    t_b, _ = stats.calibrate_threshold(human, b_list)

    pred_a = [stats.verdict_from_score(s, t_a) for s in a_list]
    pred_b = [stats.verdict_from_score(s, t_b) for s in b_list]

    acc_a = stats.confusion(human, pred_a).accuracy
    acc_b = stats.confusion(human, pred_b).accuracy
    kappa_a = stats.cohen_kappa(human, pred_a)
    kappa_b = stats.cohen_kappa(human, pred_b)

    # Judge-vs-judge agreement, treating judge-A as the reference "label".
    jj = stats.confusion(pred_a, pred_b)
    jj_agree = jj.accuracy
    jj_kappa = stats.cohen_kappa(pred_a, pred_b)

    judge_disagreements = [
        {
            "id": c["id"],
            "human": c["human_verdict"],
            "judge_a": pa,
            "judge_b": pb,
            "score_a": scores_a[c["id"]],
            "score_b": scores_b[c["id"]],
            "note": c.get("note", ""),
        }
        for c, pa, pb in zip(gold, pred_a, pred_b, strict=True)
        if pa != pb
    ]
    return {
        "n": len(gold),
        "threshold_a": t_a,
        "threshold_b": t_b,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "kappa_a": kappa_a,
        "kappa_b": kappa_b,
        "judge_agreement": jj_agree,
        "judge_kappa": jj_kappa,
        "judge_disagreements": judge_disagreements,
    }


def format_report(a: dict) -> str:
    lines = [
        "# Cross-judge spot-check: Claude (judge-A) vs OpenAI (judge-B)\n",
        f"gold cases:                 {a['n']}",
        "",
        "each judge vs the HUMAN labels (at its own calibrated threshold):",
        f"  judge-A (Claude):  acc {a['accuracy_a']:.1%}   kappa {a['kappa_a']:.2f}   "
        f"(threshold {a['threshold_a']:.2f})",
        f"  judge-B (OpenAI):  acc {a['accuracy_b']:.1%}   kappa {a['kappa_b']:.2f}   "
        f"(threshold {a['threshold_b']:.2f})",
        "",
        "judge-A vs judge-B (do the two families agree with EACH OTHER?):",
        f"  agreement:  {a['judge_agreement']:.1%}   kappa {a['judge_kappa']:.2f}",
        "",
        f"cases where the two judges disagree ({len(a['judge_disagreements'])}):",
    ]
    for d in a["judge_disagreements"]:
        lines.append(
            f"  {d['id']:28} human={d['human']:10} "
            f"A={d['judge_a']:10}({d['score_a']:.2f}) B={d['judge_b']:10}({d['score_b']:.2f}) "
            f"{d['note'][:44]}"
        )
    if not a["judge_disagreements"]:
        lines.append("  (none — the two judges reached identical verdicts on every case)")
    lines.append(
        "\nNote: a spot-check for the write-up, not a CI gate. The pinned judge and the "
        "calibrated\nthresholds are unchanged; judge-B never enters the metric path."
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-judge spot-check (Claude vs OpenAI).")
    parser.add_argument(
        "--live", action="store_true", help="call judge-B live to record (else replay cache)"
    )
    args = parser.parse_args()

    if not GOLD_PATH.exists():  # pragma: no cover - defensive
        raise SystemExit("gold set missing; expected meta_eval/gold.jsonl")
    if not args.live and not SCORES_PATH.exists():
        raise SystemExit(
            "judge-A scores missing — run `python -m meta_eval.run --live` once to record them."
        )

    try:
        scores_a = collect_scores(live=args.live)
        scores_b = collect_scores_b(live=args.live)
    except cache.CacheMiss as miss:
        raise SystemExit(
            "judge-B scores not recorded yet — run once with a key to record:\n"
            "  LIVE_LLM=1 OPENAI_API_KEY=... python -m meta_eval.cross_judge --live\n"
            "then commit evals/cache/ + meta_eval/scores_openai.json.\n"
            f"({miss})"
        ) from miss
    print(format_report(cross_analyse(scores_a, scores_b)))


if __name__ == "__main__":
    main()
