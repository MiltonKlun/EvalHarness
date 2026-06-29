"""Regression demo (plan 2.5): show the eval suite catching a broken system.

Two flavours:

1. `--mode injected` (default, quota-free): feed the metrics a deliberately *ungrounded*
   answer (as if a prompt regression made the model hallucinate) and show that the
   faithfulness metric scores it low / fails. This proves the harness CATCHES a
   regression, using zero Gemini quota — it runs the live Claude judge over a synthetic
   bad answer.

2. `--mode prompt-break` (documented, needs Gemini quota): the "real" demo from the plan
   — break app/rag.SYSTEM_PROMPT in a branch (e.g. remove the grounding instruction),
   re-run `make eval`, and watch faithfulness drop on real output. Steps printed below;
   run it when generate-quota is available (or billing is on).

    python -m evals.regression_demo
"""

from __future__ import annotations

import argparse

GROUNDED_CONTEXT = [
    "The Kestrel-2 extends flight time to 55 minutes and raises wind tolerance to 22 m/s.",
    "The Kestrel-1 has a flight time of 38 minutes and a wind tolerance of 14 m/s.",
]
# A regression (e.g. dropped grounding instruction) would let the model invent facts:
UNGROUNDED_ANSWER = (
    "The Kestrel-2 can fly for 90 minutes and tolerates winds up to 40 m/s, and it also "
    "has a built-in defibrillator for emergency rescues."
)
GROUNDED_ANSWER = "The Kestrel-2 flies for 55 minutes and tolerates winds up to 22 m/s."


def _faithfulness(answer: str) -> float:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from evals.judge import ClaudeJudge

    metric = FaithfulnessMetric(model=ClaudeJudge(), threshold=0.7)
    metric.measure(
        LLMTestCase(
            input="Tell me about the Kestrel-2's flight time and wind tolerance.",
            actual_output=answer,
            retrieval_context=GROUNDED_CONTEXT,
        )
    )
    return metric.score


def injected_demo() -> None:
    print("Regression demo (injected ungrounded answer) — live Claude judge, $0 Gemini.\n")
    good = _faithfulness(GROUNDED_ANSWER)
    bad = _faithfulness(UNGROUNDED_ANSWER)
    print(f"  grounded answer   -> faithfulness {good:.2f}  (expected high, >= 0.7)")
    print(f"  REGRESSED answer  -> faithfulness {bad:.2f}  (expected LOW, < 0.7)")
    print()
    if bad < 0.7 <= good:
        print(
            "PASS: the suite CAUGHT the regression — the ungrounded answer scored below "
            "threshold while the grounded one passed. This is the regression-catching "
            "property the whole project promises."
        )
    else:
        print("Unexpected scores — inspect the judge output above.")


def print_prompt_break_steps() -> None:
    print(
        "Prompt-break demo (real, needs Gemini generate-quota):\n"
        "  1. git checkout -b break/system-prompt\n"
        "  2. In app/rag.py, delete the grounding line from SYSTEM_PROMPT\n"
        '     ("Answer strictly from the context below. Do not use outside knowledge.")\n'
        "  3. LIVE_LLM=1 make eval   # re-records answers with the broken prompt\n"
        "  4. Observe faithfulness scores drop and answerable cases fail.\n"
        "  5. Screenshot the red run for the README, then discard the branch.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression demo.")
    parser.add_argument(
        "--mode",
        choices=["injected", "prompt-break"],
        default="injected",
        help="injected = quota-free judge demo; prompt-break = print the real-run steps",
    )
    args = parser.parse_args()
    if args.mode == "injected":
        injected_demo()
    else:
        print_prompt_break_steps()


if __name__ == "__main__":
    main()
