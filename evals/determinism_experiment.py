"""Stochasticity & determinism experiment (plan 2.6) — the standout README finding.

Samples a subset of cases N times in BOTH decode modes and reports the variance:

  - max_pinned : temperature=0 + top_p/top_k fixed + seed set (every knob the API offers)
  - near_det   : temperature=0 only

Two findings from one run:
  (a) the real run-to-run variance / hallucination behaviour of the stochastic LLM, and
  (b) that EVEN fully pinned, Gemini output still varies — temperature=0 != determinism,
      consistent with Google's own "seed is best-effort" wording.

This is a LIVE experiment (it must bypass the cache to see real variance), so it sets
LIVE_LLM and calls the model directly N times. Run it deliberately, not in CI:

    python -m evals.determinism_experiment --samples 5
"""

from __future__ import annotations

import argparse
import os
import time

# Force live mode BEFORE importing anything that reads config — variance needs real calls.
os.environ["LIVE_LLM"] = "1"

from app.rag import DECODE_MODES, _build_prompt  # noqa: E402
from app.retriever import retrieve  # noqa: E402
from shared import config, llm  # noqa: E402

# A small representative subset (keeps quota use modest).
SUBSET = [
    "Who founded Meridian Robotics?",
    "Which Meridian drone can fly in 20 m/s wind?",
    "How long is raw inspection footage retained?",
]

# Gemini's free tier caps generate at ~5 requests/MINUTE (a separate quota from the daily
# cap). This experiment fires many calls back-to-back, so without pacing it hits that RPM
# wall mid-run. Default to one call every 13s (< 5/min) so a free-tier run completes; set
# DELAY=0 with billing on to run at full speed.
_DEFAULT_DELAY_S = 13.0


def _sample(question: str, mode: str, n: int, delay_s: float) -> list[str]:
    """Call the generator n times for one question+mode, bypassing the cache."""
    chunks = retrieve(question)
    prompt = _build_prompt(question, chunks)
    params = DECODE_MODES[mode]
    model = llm._init(config.GENERATOR_MODEL, **params)
    outs = []
    for i in range(n):
        if i > 0 and delay_s:
            time.sleep(delay_s)  # stay under the free-tier per-minute cap
        content = model.invoke(prompt).content
        text = content if isinstance(content, str) else _join_text(content)
        outs.append(text.strip())
    return outs


def _join_text(blocks) -> str:
    return "\n".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    )


def run(samples: int, delay_s: float = _DEFAULT_DELAY_S) -> None:
    print(
        f"Determinism experiment — {samples} samples/case, {delay_s}s inter-call delay, "
        f"model={config.GENERATOR_MODEL}\n"
    )
    for mode in ("max_pinned", "near_det"):
        print(f"### mode = {mode}  ({DECODE_MODES[mode]})")
        for q in SUBSET:
            outs = _sample(q, mode, samples, delay_s)
            distinct = len(set(outs))
            verdict = "IDENTICAL" if distinct == 1 else f"{distinct} DISTINCT outputs"
            print(f"  [{verdict:18}] {q}")
            if distinct > 1:
                for i, o in enumerate(sorted(set(outs))):
                    print(f"      variant {i + 1}: {o[:90]}...")
        print()
    print(
        "Takeaway: any 'DISTINCT' result under max_pinned is direct evidence that "
        "temperature=0 + seed does NOT guarantee determinism on Gemini; all-IDENTICAL means "
        "this run did not surface variance (pinning held for this small factual subset)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stochasticity/determinism experiment.")
    parser.add_argument("--samples", type=int, default=5, help="samples per case per mode")
    parser.add_argument(
        "--delay",
        type=float,
        default=_DEFAULT_DELAY_S,
        help="seconds between calls (default 13s for the free-tier 5 RPM cap; 0 with billing)",
    )
    args = parser.parse_args()
    run(args.samples, args.delay)


if __name__ == "__main__":
    main()
