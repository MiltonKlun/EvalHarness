"""Experiment 3: is ARM C's 20/20 stable, or a lucky draw?

Re-runs the tuned arm 3x over the full gold set. Jev is a model; nothing guarantees
identical answers across calls. If accuracy wobbles, the 100% is a point estimate,
not a property -- and this repo of all repos should say so.
"""

import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

REPO = str(Path(__file__).resolve().parents[2])
SCRATCH = str(Path(__file__).resolve().parent)
load_dotenv(REPO + "/.env")

from typesafe_sdk import Choice, TypeSafeClient  # noqa: E402

MODEL = "jev-1.13.0"
MAP = {"supports": "grounded", "contradicts": "ungrounded", "says_nothing": "ungrounded"}
RUNS = 3

INSTR = (
    "The `answer` was produced by a system that was told to answer ONLY from `context`. "
    "Judge how `context` relates to the factual claims made in `answer`. "
    "Judge ONLY against `context` -- your own knowledge of the world is irrelevant."
)
CRIT = {
    "supports": (
        "Every factual claim in `answer` is stated in `context`, directly implied by it, "
        "or is a faithful paraphrase of it. Omitting some of the context is still supports. "
        "Abstaining when `context` really does not contain it is also supports."
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


def main() -> None:
    with open(REPO + "/meta_eval/gold.jsonl", encoding="utf-8") as fh:
        gold = [json.loads(line) for line in fh]

    client = TypeSafeClient()
    per_case = defaultdict(list)
    accs = []

    for run in range(1, RUNS + 1):
        ok = 0
        for case in gold:
            resp = client.system_one(
                {"context": case["context"], "answer": case["answer"]},
                {"relation": Choice(instructions=INSTR, criteria=CRIT)},
                model=MODEL,
            )
            a = resp.choices["relation"]
            per_case[case["id"]].append((a.choice, round(a.confidence, 3)))
            ok += MAP[a.choice] == case["human_verdict"]
        acc = ok / len(gold)
        accs.append(acc)
        print(f"run {run}: accuracy {acc:.0%}  ({ok}/{len(gold)})")

    print(f"\naccuracy across {RUNS} runs: {[f'{a:.0%}' for a in accs]}")

    print("\n=== per-case stability (option choice across runs) ===")
    unstable = 0
    for cid, obs in per_case.items():
        choices = {o[0] for o in obs}
        if len(choices) > 1:
            unstable += 1
            print(f"  UNSTABLE {cid:38} {[o[0] for o in obs]}")
    print(f"cases with a changed OPTION across {RUNS} runs: {unstable}/{len(per_case)}")

    print("\n=== confidence variation (same option, different number) ===")
    varied = 0
    for cid, obs in per_case.items():
        confs = {o[1] for o in obs}
        if len({o[0] for o in obs}) == 1 and len(confs) > 1:
            varied += 1
            print(f"  {cid:38} confs={sorted(confs)}")
    print(f"cases with identical option but varying confidence: {varied}/{len(per_case)}")

    with open(SCRATCH + "/exp3_raw.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in per_case.items()}, f, indent=2)


if __name__ == "__main__":
    main()
