# LLM Test Harness

<!-- Badges work once the repo is pushed to GitHub. Replace OWNER with your GitHub user. -->
[![CI (fast)](https://github.com/OWNER/llm-test-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/llm-test-harness/actions/workflows/ci.yml)
[![Live drift eval](https://github.com/OWNER/llm-test-harness/actions/workflows/live-eval.yml/badge.svg)](https://github.com/OWNER/llm-test-harness/actions/workflows/live-eval.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CI that catches LLM regressions — for a non-deterministic RAG agent.**

This project treats LLM behaviour as something you can *regression-test*, not just
eyeball. The system under test (a small RAG agent on **Google Gemini**) is deliberately
boring — the **tests are the product**. The evaluator is **Anthropic Claude**, a
*different model family* from the generator, so the judge never grades its own homework.

> 🚧 **Status:** Phases 0–3 complete (foundation, RAG app, functional eval suite,
> two-tier+ CI). Adversarial, agent-reliability, and meta-eval suites land in later
> phases — see [LLM_Eval_Harness_Build_Plan.md](LLM_Eval_Harness_Build_Plan.md).

## Why testing AI differs from traditional assertions

*(placeholder — expanded in Phase 7)* Traditional tests assert exact outputs. LLM
outputs are stochastic: the same prompt at `temperature=0` can still vary, hallucinate,
or drift as the model changes underneath you. This harness measures *fuzzy* qualities
(groundedness, relevancy, hallucination, safety) with calibrated thresholds, and
separates "did my code regress?" from "did the model drift?" via a two-tier CI design.

## Architecture

*(diagram placeholder — added in Phase 7: app under test (Gemini) ← suites ← Claude
judge ← two-tier CI)*

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (`irm https://astral.sh/uv/install.ps1 | iex`
on Windows, or `curl -LsSf https://astral.sh/uv/install.sh | sh` elsewhere).

```bash
# 1. Create the virtual environment and install everything (incl. dev tools)
make install            # == uv venv && uv pip install -e ".[dev]"

# 2. (optional) configure secrets — NOT needed for the offline test suite
cp .env.example .env    # then fill in keys

# 3. Lint and run the offline tests (no API keys required)
make lint
make test
```

`make help` lists all targets.

### Building the vector store (one-time, needs a key)

Retrieval runs against a committed FAISS store. It's built once from the corpus with a
real `GOOGLE_API_KEY`, then committed so every clone/CI uses identical embeddings:

```bash
python -m app.ingest --stats   # preview chunking (no key needed)
python -m app.ingest           # build + persist app/vectorstore/ (needs GOOGLE_API_KEY)
python -m app.agent "Which drone can fly in 20 m/s wind?"   # run the agent end-to-end
```

The offline test suite (`make test`) needs no keys and no store.

## The CI philosophy: replay vs. live, and why the judge isn't a merge gate

The hard problem in testing LLMs is that the same prompt gives different answers, the
model drifts under you, and the *judge* is itself a non-deterministic, paid model. The CI
is split into **three tiers** so each answers a different question with exactly the inputs
it needs:

| Tier | Trigger | Keys | What runs | What it proves |
|---|---|---|---|---|
| **Fast** (`ci.yml`) | every push / PR | **none** | lint + offline tests + the **deterministic** eval checks (abstention, citations, key-facts), replaying recorded Gemini answers *and* recorded retrieval | "my code/prompts didn't regress" — fast, free, blocking |
| **Judged** (`judged-eval.yml`) | manual / PR label | Anthropic | + the Claude-judged faithfulness & relevancy metrics, over the same recordings | "the metric logic still holds" — **non-blocking on purpose** |
| **Live drift** (`live-eval.yml`) | weekly cron + manual | Google + Anthropic | the full suite with **real** Gemini calls + the stochasticity/determinism probe | "the model didn't drift" — surfaces real variance |

**Why the judge is deliberately off the blocking path:** the Claude judge is
non-deterministic (it once flagged a perfectly correct answer as a fail) *and* costs
money. Gating every merge on a flaky paid check would produce spurious red builds, so
faithfulness/relevancy run in the opt-in judged tier and the scheduled live tier — never
as a required PR check. That's intentional CI hygiene, not a gap.

**Nothing is hardcoded.** The fast tier replays *recorded inputs* (both the Gemini answer
and the retrieved chunks) but the deterministic metrics run live every time. The real,
stochastic model is genuinely exercised — in the live tier, which is where drift belongs.

## License

[MIT](LICENSE)
