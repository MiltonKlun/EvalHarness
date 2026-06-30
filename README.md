# EvalHarness

[![CI (fast)](https://github.com/MiltonKlun/EvalHarness/actions/workflows/ci.yml/badge.svg)](https://github.com/MiltonKlun/EvalHarness/actions/workflows/ci.yml)
[![Live drift eval](https://github.com/MiltonKlun/EvalHarness/actions/workflows/live-eval.yml/badge.svg)](https://github.com/MiltonKlun/EvalHarness/actions/workflows/live-eval.yml)
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

## Testing the agent, not just the output (`make agent-tests`)

The three suites above grade the agent's *final text*. The agent-reliability suite grades
its **behaviour as a graph** — the rare skill:

- **Tool-call correctness** — did it call the right tool with the right args?
- **Loop / termination safety** — a max-step guard; a model that never stops still halts.
- **State integrity** — messages and the step counter carry correctly across steps.
- **Failure recovery** — a tool that raises is handled (becomes a recoverable result),
  not a crash. *(This caught a real defect — see [VULN-002](adversarial/FINDINGS.md).)*

These assert on **LangGraph's in-memory intermediate steps**, driven by a *scripted fake
model* — so the whole suite runs with **no API key and no network** on a fresh clone. A
LangSmith key, if present, unlocks an additional "assert against the real tracing platform"
path; without it, that one test skips cleanly. The in-memory path is always the primary
way to test — LangSmith is a bonus, never a dependency.

## How good is the judge? (meta-eval — `make meta-eval`)

We use Claude as an LLM judge for faithfulness/relevancy — so the obvious question is:
**how do we know the judge is right?** We measure it against a **hand-labeled gold set** of
20 (answer, context, verdict) cases, including deliberately tricky ones.

Result on the current judge (Claude Haiku):

| metric | value |
|---|---|
| accuracy vs. human labels | **80%** |
| Cohen's κ (chance-corrected) | **0.60** (substantial) |
| error direction | **4 false positives, 0 false negatives** |

The errors aren't random — they're a **systematic lenient bias**: the judge marks an answer
faithful when the added claim is *plausible or true in the real world*, even though it isn't
in the provided context (e.g. "Aberdeen is in the UK" — true, but not in the documents). So
**our faithfulness scores are an upper bound on true groundedness**, and we say so. Full
write-up: [JUDGE-001 in FINDINGS.md](adversarial/FINDINGS.md).

What we do about it: (1) the metric **threshold is calibrated** against this gold set, not
guessed (see [thresholds.yaml](thresholds.yaml)); (2) a **0.05 margin** keeps us off the
judge's noisy boundary; (3) the deterministic checks catch several of these independently;
(4) **`make meta-eval` is also the judge-drift detector** — re-running it re-measures
agreement, and if a future Claude version degrades, the suite goes red.

This is the difference between "I used an eval framework" and "I measured my evaluator and
know exactly where it's wrong."

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

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Milton Klun**  
*QA Automation Engineer | AI Quality Testing*

<div align="left">
  <a href="https://www.linkedin.com/in/milton-klun/"><img src="https://img.shields.io/badge/LINKEDIN-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a><a href="mailto:miltonericklun@gmail.com"><img src="https://img.shields.io/badge/EMAIL-D14836?style=for-the-badge" alt="Email"/></a><a href="https://www.miltonklun.com"><img src="https://img.shields.io/badge/PORTFOLIO-000000?style=for-the-badge" alt="Live Site"/></a>
</div>
