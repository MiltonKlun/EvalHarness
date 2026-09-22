"""Keyless: tally the DeepEval verdicts already committed in evals/cache/.

Backs the functional-suite figures in docs/typesafe-jev-evaluation.md §8. No API calls.
DeepEval returns both faithfulness and answer-relevancy verdicts under a schema named
"Verdicts", so calls are split by prompt: faithfulness prompts carry "Retrieval Contexts".

    .venv/Scripts/python.exe docs/jev-evidence/analyze_cached_verdicts.py
"""

import json
from collections import Counter
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "evals" / "cache"


def score(verdicts: list[str], penalize_idk: bool) -> float:
    """DeepEval's faithfulness _calculate_score, reproduced exactly."""
    if not verdicts:
        return 1.0
    n = sum(v != "no" for v in verdicts)
    if penalize_idk:
        n -= sum(v == "idk" for v in verdicts)
    return n / len(verdicts)


def main() -> None:
    tallies = {"faithfulness": Counter(), "relevancy": Counter()}
    faith_idk = []
    for path in sorted(CACHE.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("params", {}).get("schema") != "Verdicts":
            continue
        kind = "faithfulness" if "Retrieval Contexts" in rec["prompt"] else "relevancy"
        verdicts = json.loads(rec["response"])["verdicts"]
        vs = [v["verdict"].strip().lower() for v in verdicts]
        tallies[kind].update(vs)
        if kind == "faithfulness" and "idk" in vs:
            faith_idk.append((vs, [v.get("reason") for v in verdicts if v.get("reason")]))

    for kind, tally in tallies.items():
        print(f"{kind:13} verdicts: {dict(tally)}")
    print(f"\nfaithfulness calls containing idk: {len(faith_idk)}")
    for vs, reasons in faith_idk:
        print(f"  default={score(vs, False):.2f}  penalize_idk={score(vs, True):.2f}  {vs}")
        for r in reasons:
            print(f"    reason: {r[:150]}")


if __name__ == "__main__":
    main()
