"""Experiment 2: is the 20/20 real, or did I tune the prompt to the test?

My exp1 criteria named the JUDGE-001 failure mode almost explicitly ("TRUE IN THE REAL
WORLD but absent", "even if `answer` first acknowledges the information is missing").
That is arguably teaching to the test. This re-runs the 4 decisive cases under
progressively WEAKER criteria to see how much of the result survives.

ARM A: bare option names, no descriptions at all (criteria values = None)
ARM B: generic cookbook-style descriptions, no mention of the failure mode
ARM C: exp1's tuned criteria (the control)
"""

import json
from pathlib import Path

from dotenv import load_dotenv

REPO = str(Path(__file__).resolve().parents[2])
SCRATCH = str(Path(__file__).resolve().parent)
load_dotenv(REPO + "/.env")

from typesafe_sdk import Choice, TypeSafeClient  # noqa: E402

MODEL = "jev-1.13.0"
MAP = {"supports": "grounded", "contradicts": "ungrounded", "says_nothing": "ungrounded"}

GENERIC_INSTR = "How does the `context` relate to the claims made in the `answer`?"

ARMS = {
    "A_bare": (
        GENERIC_INSTR,
        {"supports": None, "contradicts": None, "says_nothing": None},
    ),
    "B_generic": (
        GENERIC_INSTR,
        {
            "supports": "The context states or directly implies the claim is true.",
            "contradicts": "The context states the opposite, or implies the claim is false.",
            "says_nothing": "The context does not address the claim either way.",
        },
    ),
    "C_tuned": (
        (
            "The `answer` was produced by a system that was told to answer ONLY from "
            "`context`. Judge how `context` relates to the factual claims made in `answer`. "
            "Judge ONLY against `context` -- your own knowledge of the world is irrelevant."
        ),
        {
            "supports": (
                "Every factual claim in `answer` is stated in `context`, directly implied "
                "by it, or is a faithful paraphrase of it. Omitting some of the context is "
                "still supports. Abstaining when `context` really does not contain it is "
                "also supports."
            ),
            "contradicts": (
                "`answer` states something that conflicts with `context` -- a different "
                "number, a different fact, or the opposite of what `context` says."
            ),
            "says_nothing": (
                "`answer` asserts something that `context` does not address either way. "
                "This INCLUDES a claim that is TRUE IN THE REAL WORLD but simply absent "
                "from `context`, and it includes invented specifics, added details, and "
                "guesses -- even plausible ones, and even if `answer` first acknowledges "
                "the information is missing."
            ),
        },
    ),
}


def main() -> None:
    with open(REPO + "/meta_eval/gold.jsonl", encoding="utf-8") as fh:
        gold = [json.loads(line) for line in fh]

    client = TypeSafeClient()
    results = {}

    for arm, (instr, crit) in ARMS.items():
        print(f"\n=== ARM {arm} ===")
        preds, rows = [], []
        for case in gold:
            resp = client.system_one(
                {"context": case["context"], "answer": case["answer"]},
                {"relation": Choice(instructions=instr, criteria=crit)},
                model=MODEL,
            )
            a = resp.choices["relation"]
            preds.append(MAP[a.choice])
            rows.append(
                {
                    "id": case["id"],
                    "human": case["human_verdict"],
                    "choice": a.choice,
                    "confidence": a.confidence,
                }
            )
        acc = sum(p == c["human_verdict"] for p, c in zip(preds, gold, strict=True)) / len(gold)
        wrong = [
            (r["id"], r["human"], r["choice"])
            for r, p in zip(rows, preds, strict=True)
            if p != r["human"]
        ]
        print(f"accuracy {acc:.0%}   errors: {len(wrong)}")
        for w in wrong:
            print(f"   MISS {w[0]:38} human={w[1]:10} jev={w[2]}")
        results[arm] = rows

    with open(SCRATCH + "/exp2_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nwrote exp2_raw.json")


if __name__ == "__main__":
    main()
