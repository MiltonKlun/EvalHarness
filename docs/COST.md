# Cost & quota budget

This harness is built to run its everyday work **for free, with no API keys** — the fast CI
tier and the whole offline test suite replay recorded responses. Real spend happens only in
the **live tier** (drift detection), and even there the suites are sized to stay inside the
free tiers of both providers. This doc shows the math so the budget is auditable, not assumed.

## The two models, and why these snapshots

| role | model (pinned) | why this one |
|---|---|---|
| Generator (system under test) | `google_genai:gemini-2.5-flash` | Chosen over the 3.x flash because 2.5-flash has the more workable free-tier `generate_content` limit. **Measured live: ~20 generate-requests/day** on the AI Studio free tier (the API's own 429 reports `limit: 20`). That hard cap is exactly why record/replay matters — see below. |
| Judge (evaluator) | `anthropic:claude-haiku-4-5` | Cheapest current Claude; a deliberately **different family** from the generator. |
| Embeddings | `gemini-embedding-2` | Used **once** to build the committed FAISS store, then never again per-run. |

Snapshots are pinned in [`shared/config.py`](../shared/config.py) — floating aliases would
let the model drift silently, which is the whole thing this project tests against.

## Where requests come from

The fast tier makes **zero** API calls (pure replay). The **judged offline replay is now
keyless too**: Claude judge verdicts are recorded like every other LLM call, so `make test`
and the fast tier replay them with no Anthropic key — a fresh clone runs the *full* judged
suite for free. Live judge spend happens only when you deliberately re-run against a live
Claude (the judged tier's fresh-call check, and the live tier). Costs below are **per full
live-tier run** ([`live-eval.yml`](../.github/workflows/live-eval.yml)). Suite sizes are the
real counts in the repo today:

| suite | cases | Gemini calls | Claude (judge) calls | notes |
|---|---:|---:|---:|---|
| Functional eval | 21 | 21 | ~32 | 1 generate per case; the judge runs only on the 16 *answerable* cases (12 answerable + 4 multihop) × 2 metrics — the 5 unanswerable cases never call the judge |
| Adversarial red-team | 16 | 16 | 4 | 4 payload sets × 4 cases; only the 4 toxicity cases use the Claude judge |
| Agent reliability (live tool-choice) | ~3 | ~3 | 0 | only the live tool-selection check hits Gemini; the rest is keyless/in-memory |
| Meta-eval (judge-drift) | 20 | 0 | 20 | judge-only — re-grades the gold set, no generator calls |
| Determinism probe | 3 subset | 18 | 0 | 3 questions × 2 decode modes × 3 samples (default) |
| **Per live run (approx.)** | | **~58 Gemini** | **~56 Claude** | |

### One-time, not per-run

- **Vector store build** (`python -m app.ingest`): one embedding pass over the corpus, then the
  FAISS store is **committed to the repo**. Every clone and CI run reuses identical embeddings —
  zero embedding calls per run.

## Against the free tiers

- **Gemini 2.5-flash free tier ≈ 20 generate-requests/day** (measured — the 429 reports
  `limit: 20`). A *full* live run is **~58 generate requests**, so it **exceeds the daily cap**
  and must be spread across days, or run with billing enabled. **This is the central cost fact
  of the project, and the reason record/replay exists:** you pay for those ~58 stochastic calls
  *once*, commit the recordings, and then re-run the (free, keyless) metric code over them on
  every PR forever. The expensive tier is rare and quota-bounded by design; the cheap tier is
  unlimited.
- **In practice you don't run the whole suite live at once.** You record a baseline incrementally
  (e.g. the 13-case adversarial re-record after a prompt change fits comfortably in two daily
  windows), and the weekly live cron tolerates a partial run — its steps are `continue-on-error`,
  so a mid-run 429 uploads what it got instead of failing.
- **Claude Haiku** judge calls (~66/run) are a few cents at Haiku pricing — negligible, and the
  judged work is opt-in / scheduled, never on the blocking PR path. Claude quota was *not* the
  binding constraint in any run; Gemini's 20/day always is.

## Behaviour at the limit (429 / quota)

The live tier is **best-effort by design** — a quota cap is a real-world condition to handle
gracefully, not a build failure:

1. **Transient errors** (503/overload, short 429s) are **retried with backoff** in
   [`shared/llm.py`](../shared/llm.py) (`_with_retries`) so a blip doesn't kill a recording run.
2. **A sustained daily-cap 429** surfaces after the retries — that's a genuine limit, not a
   blip — and the live job's steps run with **`continue-on-error: true`**, so the scheduled run
   uploads whatever it produced instead of going red on quota.
3. **Fresh re-recordings and judge scores are uploaded as a run artifact, not auto-committed**
   — the live steps' fresh `evals/cache/` and `meta_eval/scores.json` land in the
   `live-eval-reports` artifact so drift can be diffed against the committed baseline and
   reviewed by a human before the baseline is updated. `scores.json` is deliberately never
   auto-committed (it is the calibration baseline). The **one** thing the live tier commits
   back automatically is the metrics-over-time row (`evals/history/runs.csv`) — and only when
   the run's metrics actually changed, so a flat trend never spawns a weekly noise commit.

> ⚠️ **Re-recording cost.** Changing the system prompt invalidates every cached agent response
> (the cache keys on the prompt), forcing a fresh re-record of the functional **and**
> adversarial baselines (~37 generate calls). At ~20/day that **spans roughly two daily quota
> windows** — which is exactly what happened for the VULN-001 hardening (verified cases first,
> the remaining 13 on the next reset). It's the one operation that spends real Gemini quota
> deliberately — see the re-record note in
> [adversarial/FINDINGS.md](../adversarial/FINDINGS.md) (VULN-001).

## The bottom line

- **Day-to-day development and every PR: $0, no keys** — fast CI and `make test` replay
  (including the judge verdicts, which are now recorded too).
- **A full live drift run (~58 Gemini + ~56 Claude calls)** exceeds the free Gemini 20/day cap,
  so on the free tier it's run rarely or incrementally (`make record-missing` pays only for
  what changed). Claude judge cost is a few cents.
- **With billing enabled** (the maintainer's operative mode) the daily cap disappears and a full
  live run costs on the order of **$0.05**. The free-tier math above stays as the documented
  baseline because the constraint is part of the story — record/replay is precisely the design
  that makes the harness usable *without* billing.
- The design choice that makes the day-to-day free is **record/replay**: pay for the stochastic
  model once, commit the recordings, then re-run the free, deterministic metric *code* over them
  as many times as you like.
