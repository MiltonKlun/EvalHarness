# Cost & quota budget

This harness is built to run its everyday work **for free, with no API keys** — the fast CI
tier and the whole offline test suite replay recorded responses. Real spend happens only in
the **live tier** (drift detection), and even there the suites are sized to stay inside the
free tiers of both providers. This doc shows the math so the budget is auditable, not assumed.

## The two models, and why these snapshots

| role | model (pinned) | why this one |
|---|---|---|
| Generator (system under test) | `google_genai:gemini-2.5-flash` | Free AI Studio tier gives **~1,500 requests/day**. The newer 3.x flash is throttled to ~20/day — unusable for a request-hungry eval harness. |
| Judge (evaluator) | `anthropic:claude-haiku-4-5` | Cheapest current Claude; a deliberately **different family** from the generator. |
| Embeddings | `gemini-embedding-2` | Used **once** to build the committed FAISS store, then never again per-run. |

Snapshots are pinned in [`shared/config.py`](../shared/config.py) — floating aliases would
let the model drift silently, which is the whole thing this project tests against.

## Where requests come from

The fast tier makes **zero** API calls (pure replay). Costs below are **per full live-tier
run** ([`live-eval.yml`](../.github/workflows/live-eval.yml)). Suite sizes are the real counts
in the repo today:

| suite | cases | Gemini calls | Claude (judge) calls | notes |
|---|---:|---:|---:|---|
| Functional eval | 21 | 21 | ~42 | 1 generate per case; judge runs faithfulness **+** relevancy (~2 judge calls/case) |
| Adversarial red-team | 16 | 16 | 4 | 4 payload sets × 4 cases; only the 4 toxicity cases use the Claude judge |
| Agent reliability (live tool-choice) | ~3 | ~3 | 0 | only the live tool-selection check hits Gemini; the rest is keyless/in-memory |
| Meta-eval (judge-drift) | 20 | 0 | 20 | judge-only — re-grades the gold set, no generator calls |
| Determinism probe | 3 subset | 18 | 0 | 3 questions × 2 decode modes × 3 samples (default) |
| **Per live run (approx.)** | | **~58 Gemini** | **~66 Claude** | |

### One-time, not per-run

- **Vector store build** (`python -m app.ingest`): one embedding pass over the corpus, then the
  FAISS store is **committed to the repo**. Every clone and CI run reuses identical embeddings —
  zero embedding calls per run.

## Against the free tiers

- **Gemini 2.5-flash free tier ≈ 1,500 requests/day.** A full live run is **~58 generate
  requests** — under **4%** of the daily cap. Even the weekly cron plus a few manual runs stays
  comfortably free.
- **Per-minute limits** are the more likely pinch than the daily cap; the suites are small and
  serial, and the live job tolerates the occasional 429 (below).
- **Claude Haiku** judge calls (~66/run) are a few cents at Haiku pricing — negligible, and the
  judged work is opt-in / scheduled, never on the blocking PR path.

## Behaviour at the limit (429 / quota)

The live tier is **best-effort by design** — a quota cap is a real-world condition to handle
gracefully, not a build failure:

1. **Transient errors** (503/overload, short 429s) are **retried with backoff** in
   [`shared/llm.py`](../shared/llm.py) (`_with_retries`) so a blip doesn't kill a recording run.
2. **A sustained daily-cap 429** surfaces after the retries — that's a genuine limit, not a
   blip — and the live job's steps run with **`continue-on-error: true`**, so the scheduled run
   uploads whatever it produced instead of going red on quota.
3. **Re-recordings are an artifact, not auto-committed** — drift is reviewed before the baseline
   is updated.

> ⚠️ **Re-recording cost.** Changing the system prompt invalidates every cached agent response
> (the cache keys on the prompt), forcing a fresh re-record of the functional **and**
> adversarial baselines (~37 generate calls). That's well within the daily cap in one sitting,
> but it's the one operation that spends real quota deliberately — see the re-record note in
> [adversarial/FINDINGS.md](../adversarial/FINDINGS.md) (VULN-001).

## The bottom line

- **Day-to-day development and every PR: $0, no keys** — fast CI and `make test` replay.
- **A full live drift run: well under 4% of the free Gemini daily cap + cents of Claude.**
- The design choice that makes this true is **record/replay**: pay for the stochastic model
  once, then re-run the (free, deterministic) metric *code* over the recording as many times as
  you like.
