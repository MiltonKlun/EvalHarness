"""Experiment 1: does Jev separate the JUDGE-001 lenient-bias cases?

Scores all 20 gold cases with one three-way Choice (supports/contradicts/says_nothing),
the citation-check cookbook shape. says_nothing is the bucket a binary faithfulness
score structurally cannot express -- which is exactly the JUDGE-001 failure class.

Writes raw results to scratchpad JSON. No repo files touched.
"""

import json
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = str(Path(__file__).resolve().parents[2])
SCRATCH = str(Path(__file__).resolve().parent)

load_dotenv(REPO + "/.env")

from typesafe_sdk import Choice, TypeSafeClient  # noqa: E402

MODEL = "jev-1.13.0"

# The 4 known Claude false positives from JUDGE-001 -- the decisive subset.
KNOWN_FP = {
    "g_correct_world_fact_unsupported",
    "g_overconfident_inference",
    "g_invented_specific",
    "g_hedge_then_invent",
}

CRITERIA = {
    "supports": (
        "Every factual claim in `answer` is stated in `context`, directly implied by it, "
        "or is a faithful paraphrase of it. Omitting some of the context is still supports. "
        "Abstaining (saying the documents do not contain the answer) when `context` really "
        "does not contain it is also supports."
    ),
    "contradicts": (
        "`answer` states something that conflicts with `context` -- a different number, a "
        "different fact, or the opposite of what `context` says."
    ),
    "says_nothing": (
        "`answer` asserts something that `context` does not address either way. This "
        "INCLUDES a claim that is TRUE IN THE REAL WORLD but simply absent from `context`, "
        "and it includes invented specifics, added details, and guesses -- even plausible "
        "ones, and even if `answer` first acknowledges the information is missing."
    ),
}

INSTRUCTIONS = (
    "The `answer` was produced by a system that was told to answer ONLY from `context`. "
    "Judge how `context` relates to the factual claims made in `answer`. "
    "Judge ONLY against `context` -- your own knowledge of the world is irrelevant here."
)


def main() -> None:
    with open(REPO + "/meta_eval/gold.jsonl", encoding="utf-8") as fh:
        gold = [json.loads(line) for line in fh]

    client = TypeSafeClient()
    out = []

    for i, case in enumerate(gold, 1):
        t0 = time.time()
        resp = client.system_one(
            {"context": case["context"], "answer": case["answer"]},
            {"relation": Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)},
            model=MODEL,
        )
        ans = resp.choices["relation"]
        rec = {
            "id": case["id"],
            "human": case["human_verdict"],
            "note": case.get("note", ""),
            "choice": ans.choice,
            "probabilities": dict(ans.probabilities),
            "confidence": ans.confidence,
            "known_claude_fp": case["id"] in KNOWN_FP,
            "latency_s": round(time.time() - t0, 2),
        }
        out.append(rec)
        flag = " <-- KNOWN CLAUDE FP" if rec["known_claude_fp"] else ""
        print(
            f"{i:2}. {case['id']:38} human={case['human_verdict']:10} "
            f"jev={ans.choice:13} conf={ans.confidence:.2f}{flag}",
            flush=True,
        )

    path = SCRATCH + "/exp1_raw.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote " + path)


if __name__ == "__main__":
    main()
