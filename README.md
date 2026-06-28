# LLM Test Harness

**CI that catches LLM regressions — for a non-deterministic RAG agent.**

This project treats LLM behaviour as something you can *regression-test*, not just
eyeball. The system under test (a small RAG agent on **Google Gemini**) is deliberately
boring — the **tests are the product**. The evaluator is **Anthropic Claude**, a
*different model family* from the generator, so the judge never grades its own homework.

> 🚧 **Status:** Phase 0 (repo, tooling & record/replay foundation) complete. RAG app,
> eval suites, and CI land in later phases — see
> [LLM_Eval_Harness_Build_Plan.md](LLM_Eval_Harness_Build_Plan.md).

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

## The two-tier CI idea (in one paragraph)

The **fast tier** (every commit) does **record/replay**: it replays *recorded LLM
responses* from `evals/cache/` but re-runs the eval **metrics live** every time — so it
genuinely tests the harness and app contract offline, for free, with no keys. Nothing is
hardcoded; the recording is just an input fixture. The **live tier** (manual + weekly)
makes real Gemini + Claude calls to catch *model drift* and surface real stochastic
behaviour. See the build plan's "Two-tier CI" box for the full rationale.

## License

[MIT](LICENSE)
