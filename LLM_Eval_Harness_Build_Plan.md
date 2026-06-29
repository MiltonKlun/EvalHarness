# LLM Eval Harness — Modular Build Plan (v3)

**Stack decisions (locked):**
- **Generator (system under test):** Google Gemini API via `langchain-google-genai` (free-tier AI Studio key). Vertex AI upgrade is an optional stretch.
- **Judge (evaluator):** **Anthropic Claude** via `langchain-anthropic` — deliberately a *different model family* from the generator so the evaluator never grades its own homework. Wired through DeepEval's custom-LLM judge interface.
- **Provider abstraction:** all model access goes through `shared/llm.py` using LangChain's `init_chat_model("<provider>:<model>")`. Swapping a provider is a one-line config change; agent/tool/prompt code stays identical.
- **CI cost strategy:** two-tier — **record/replay** in fast CI (real metrics re-run live over recorded LLM responses), real Gemini+Claude calls in the live tier (manual + scheduled). See the "Two-tier CI — what it really means" box below.
- **Python:** assumed strong; tasks are atomic units of work, not syntax tutorials.

**Build order is strict:** finish and merge each phase (green CI) before starting the next. Each phase is independently shippable and already a portfolio piece.

**Legend:** `[ ]` task · `·` subtask · 🎯 phase exit criteria

---

### ⚠️ Two-tier CI — what it really means (read this once, it drives the whole design)

The fast tier does **record/replay**, NOT hardcoded golden answers.

- The **cache stores raw LLM responses only** (the expensive, non-deterministic part).
- The **eval metrics — groundedness, relevancy, hallucination, the Claude judge — re-run live, as real code, on every CI run**, on top of the replayed responses.
- This means fast CI genuinely tests *our harness, thresholds, and app contract* offline, for free, with no keys — without faking the science.

| | **Fast tier** (every commit/PR) | **Live tier** (manual `workflow_dispatch` + weekly cron) |
|---|---|---|
| Calls real Gemini/Claude? | No — replays recorded responses | Yes — real API calls |
| Speed / cost / keys | Seconds, free, no keys | Slow, uses quota, needs secrets |
| What it actually proves | "*My code/prompts/metrics* didn't regress" | "*The model* didn't drift; expose real stochastic behavior" |
| Determinism | Reproducible (replayed input + live metric code) | Intentionally non-deterministic (that's the point) |

**The honest framing for the README:** *Nothing is hardcoded.* The recording is just an **input fixture** for the cheap tier; the evaluation logic always runs live. The **real, stochastic LLM is genuinely tested in the live tier**, where we even sample each case multiple times to surface hallucinations and variance (Phase 2.6). Fast CI is a regression guard for *our* code; drift detection is exclusively the live tier's job.

---

### 🎚️ Threshold strategy — how we set, defend, and avoid gaming our pass/fail lines

*(This is the methodology a recruiter will probe hardest: "your faithfulness metric is 0.0–1.0 — why is the pass line 0.7 and not 0.85?" The plan must have an answer that isn't "it felt right.")*

**Our defensible answer, in order:**

1. **Thresholds are *calibrated*, not guessed.** Each metric's pass line is chosen against the **Phase 6 hand-labeled gold set** (the meta-eval). We pick the threshold that best separates *human-labeled pass* from *human-labeled fail* cases — i.e. we tune the cutoff to maximize agreement with human judgment, then record that number. This is why Phase 6 is not a vanity layer: **the meta-eval is what justifies the thresholds.**
2. **We separate two kinds of threshold, because they answer different questions:**
   - **Absolute gates** (e.g. groundedness on answerable cases ≥ X) → "is a single answer good enough to ship?"
   - **Aggregate/regression gates** (e.g. suite-mean must not drop > Y vs the committed baseline) → "did this commit make things *worse*?" Non-deterministic systems need the second kind; a single case dipping below an absolute line on one sample isn't necessarily a regression, but a *fleet-wide* mean drop is.
3. **We acknowledge the band of uncertainty.** Because the judge is itself a model (Phase 6 measures its error rate), a threshold isn't infinitely precise. We set gates with a deliberate **margin** away from the judge's known disagreement zone, rather than pretending 0.001 differences are meaningful.
4. **We resist Goodhart / gaming.** The eval dataset (2.1) is hand-authored and independent of the model under test; the judge (Claude) is a different family from the generator (Gemini). So thresholds can't be satisfied by the generator "teaching to its own test."
5. **Thresholds are versioned config, not magic numbers in code.** They live in one place (`shared/config.py` / a `thresholds.yaml`), with a comment citing the gold-set calibration run that produced each one. Changing a threshold is a reviewable diff with a rationale.

**One-line interview soundbite:** *"My thresholds are calibrated against a human-labeled gold set, expressed as both per-answer gates and baseline-relative regression gates, with a margin sized to the judge's measured error rate — so they're defensible, not vibes."*

---

## Phase 0 — Repo, Tooling & Replay Foundation
*Goal: a clean, reproducible skeleton — including the record/replay cache — before any eval code.*

- [ ] **0.1 Initialize repo**
  - · `git init`, public GitHub repo `EvalHarness`, MIT license, `.gitignore` (Python + `.env`)
  - · Folder skeleton: `app/ evals/ adversarial/ agent_tests/ meta_eval/ shared/ .github/workflows/`
- [ ] **0.2 Python environment**
  - · `pyproject.toml` (**uv**), pin Python 3.11+
  - · **Create + use a virtual environment** (`uv venv` → `.venv/`, gitignored). All commands run inside it (`uv run …`); document the activate/`make install` step in the README quickstart so a recruiter cloning the repo reproduces the exact env
  - · Core deps: `langchain`, `langchain-google-genai`, `langchain-anthropic`, `langgraph`, `langsmith`, `deepeval`, `pytest`, `python-dotenv`, `pyyaml` (for `thresholds.yaml`)
- [ ] **0.3 Secrets & config**
  - · `.env.example` listing `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LIVE_LLM` (default `0`)
  - · `shared/config.py` to load + validate env vars, fail loud if missing
- [ ] **0.4 Provider abstraction (`shared/llm.py`)**
  - · Thin wrapper over `init_chat_model("google_genai:<pinned-model>")` for the generator and `init_chat_model("anthropic:<pinned-model>")` for the judge
  - · **Pin exact model snapshot strings** (not floating aliases) in config — both generator and judge
- [ ] **0.5 Record/replay cache (foundational — used by every later phase)**
  - · `shared/cache.py`: key = hash(prompt + model + params + provider) → recorded raw LLM response on disk (`evals/cache/`)
  - · `LIVE_LLM=1` → real call **and record**; default (`0`) → replay from cache, **hard-fail on a cache miss** (a miss in fast CI must error, never silently call the API or pass)
  - · This lives in Phase 0 so Phase 1 onward is reproducible from the very first LLM call
- [ ] **0.6 Pre-commit hygiene & task runner**
  - · `ruff` (lint+format), `pre-commit` hook, `make`/`task` runner with `lint`, `test`, `eval`, `eval-ci` targets (more added per phase)
- [ ] **0.7 README stub**
  - · Title, one-paragraph pitch, "why AI testing differs from traditional assertions" placeholder, architecture diagram placeholder

🎯 **Exit:** repo clones, installs clean, `make lint` passes, secrets documented but not committed, `shared/cache.py` can record and replay a trivial call.

---

## Phase 1 — The System Under Test (a *correct* boring RAG)
*Goal: a minimal but properly-built RAG agent. Small on purpose — the tests are the star — but the retrieval must be sound, or every downstream "groundedness" number is noise.*

- [ ] **1.1 Corpus**
  - · Pick a small fixed document set (10–30 docs) where "correct grounded answer" is checkable
  - · Store raw docs in `app/corpus/`, document the source + license in README
- [ ] **1.2 Ingestion + retrieval (pin everything that can drift)**
  - · Chunking with an explicit, documented strategy (size + overlap) — record the choice; it affects groundedness
  - · Embeddings via `GoogleGenerativeAIEmbeddings` with a **pinned embedding-model version**
  - · Local vector store (Chroma/FAISS) **persisted to disk AND committed to the repo** (or built by a deterministic, cached fixture step) — so retrieval is identical across machines/CI and embedding drift can't silently move results
  - · `app/retriever.py` with a clean `retrieve(query) -> list[Chunk]` interface
  - · ⚠️ RAG-correctness checklist (do not skip): chunk boundaries don't split key facts; top-k is tuned (not left at default); retrieved chunks are deduplicated; `retrieve()` returns the source metadata needed later for citation/groundedness checks
- [ ] **1.3 RAG chain**
  - · `app/rag.py`: prompt template + Gemini call (via `shared/llm.py`), returns `{answer, contexts, sources}`
  - · System prompt instructs grounding **and explicit abstention** ("if the corpus doesn't contain the answer, say so") — this is what the unanswerable test cases will probe
  - · Two configurable decode modes (set up here, exercised in 2.6): **(a) max-pinned mode** = pinned model snapshot + `temperature=0` + fixed `top_p`/`top_k` + `seed` set, and **(b) near-deterministic mode** = `temperature=0` only
  - · ⚠️ **Honest statement to bake into the README (see 2.6):** Gemini *does* expose a `seed`, but Google's own docs state it is **best-effort, not guaranteed** — "deterministic output isn't guaranteed," and a temp=0 response is only "*mostly* deterministic." We therefore **do not claim to achieve determinism**; we pin every knob the API offers and then *measure the residual variance*. This is a deliberate intellectual-honesty decision, and the finding (variance persists even fully pinned) is more credible than pretending we forced determinism.
- [ ] **1.4 LangSmith tracing on**
  - · Verify runs appear in LangSmith; tag runs with a project name
- [ ] **1.5 Wrap as LangGraph agent**
  - · `app/agent.py`: a graph with a retrieval tool + one external tool (stub or simple API)
  - · Single happy-path manual test that the graph runs end to end (reproducible via the Phase-0 cache)

🎯 **Exit:** `python -m app.agent "question"` returns a grounded answer; vector store is committed/reproducible; embedding + model versions pinned; trace visible in LangSmith.

---

## Phase 2 — Suite 1: Functional Evals (Project 1 core)
*Goal: the flagship eval layer. Build this fully green before anything else.*

- [ ] **2.1 Versioned eval dataset**
  - · `evals/dataset.jsonl`: question → reference answer + expected source/context
  - · 15–25 cases incl. edge cases with **defined pass conditions**:
    - · *unanswerable-from-corpus* → pass = model abstains, does not invent
    - · *multi-hop* → pass = answer is grounded AND the expected source set includes **all** the chunks required to reach it (not just one), verifiable from `contexts`/`sources`
    - · (*ambiguous* cases deliberately **excluded** — no objective pass condition, low signal-to-noise; documenting this exclusion is itself a sign of eval discipline)
  - · **Provenance / independence:** cases are **hand-authored from the corpus by a human, NOT generated by Gemini** — so the eval set is independent of the system under test and can't be "taught to its own test" (same independence principle as the Claude judge)
  - · Push dataset to LangSmith as a named dataset (shows real eval-tooling use)
- [ ] **2.2 Metric assertions (DeepEval, judged by Claude)**
  - · Configure DeepEval's judge as the **Claude** model from `shared/llm.py` (custom-LLM judge interface) — *generator = Gemini, judge = Claude*, documented as an independence property
  - · Groundedness / faithfulness metric
  - · Answer relevancy metric
  - · Format/schema compliance check (output shape, citations present) — note: this is a **deterministic, non-LLM** assertion, the cheap first line of defense before any judge runs
  - · Hallucination check on the unanswerable cases (must abstain, not invent)
  - · **Thresholds** are read from versioned config (`thresholds.yaml` / `shared/config.py`), each calibrated against the Phase 6 gold set (see Threshold-strategy box). Initial provisional values are placeholders until 6.2 runs; the calibration step replaces them.
  - · **Judge-failure handling (operational maturity):** if a Claude judge call errors / times out / rate-limits, the metric does **not** silently pass — it (a) retries with backoff, then (b) marks the case `JUDGE_ERROR` (distinct from pass/fail) and surfaces it in the report; a run with any `JUDGE_ERROR` is flagged, not green
- [ ] **2.3 Record/replay wiring (reuse Phase 0 cache)**
  - · Confirm each dataset case records its raw Gemini (and any Claude) responses to `evals/cache/`
  - · Commit a baseline recording so fast CI runs offline with zero API calls — **but metrics still run live over those recordings**
- [ ] **2.4 Test runner**
  - · `pytest` wiring so each dataset case is a parametrized test
  - · `make eval` (live: real calls + re-record) vs `make eval-ci` (replay + live metrics) targets
- [ ] **2.5 Baseline + regression demo**
  - · Capture a green baseline; intentionally break the system prompt in a branch to show fast CI catching the regression via *live metrics over recorded inputs* (screenshot/log for README)
- [ ] **2.6 Stochasticity & determinism experiment (live tier) — one combined study**
  - · In live mode, sample each (or a subset of) case **N times** in **both** decode modes from 1.3 (max-pinned vs near-deterministic), and report the score distribution / variance, not a single pass/fail
  - · Two findings from one experiment: (a) the real **hallucination rate + run-to-run variance** of the stochastic LLM — the "real test" the project promises; and (b) a concrete demonstration that **even fully pinned (`temperature=0` + `seed` + `top_p`/`top_k`), Gemini output still varies** — i.e. `temperature=0 ≠ determinism`, consistent with Google's own "best-effort" wording
  - · This becomes a standout README section; its variance numbers also feed the threshold *margin* decision (Threshold-strategy box, point 3)

🎯 **Exit:** `make eval-ci` runs offline and green (metrics live, inputs replayed); live run works; one documented caught regression; one documented stochasticity/determinism finding.

---

## Phase 3 — CI Integration (two-tier)
*Goal: automation that proves regression-catching for non-deterministic systems.*

- [ ] **3.1 Fast CI workflow (every commit/PR)**
  - · `.github/workflows/ci.yml`: lint + `make eval-ci` (replay inputs, live metrics, no keys needed)
  - · Fails the build on metric threshold breach OR on any cache miss
- [ ] **3.2 Live CI workflow (manual + scheduled)**
  - · `.github/workflows/live-eval.yml`: `workflow_dispatch` + weekly `cron`
  - · Uses `GOOGLE_API_KEY` + `ANTHROPIC_API_KEY` GitHub secrets; runs real calls to detect model drift; runs the 2.6 multi-sample probe
  - · Uploads results as an artifact / posts summary
- [ ] **3.3 Status surface**
  - · CI badge in README; short note on the two-tier (record/replay vs live-drift) philosophy

🎯 **Exit:** PRs run replayed evals automatically; live drift check runnable on demand; badges green.

---

## Phase 4 — Suite 2: Adversarial / Red-Team (Project 2)
*Goal: a documented attack catalog + automated runner.*

- [ ] **4.1 Attack catalog (the documented artifact)**
  - · `adversarial/CATALOG.md`: categories — prompt injection, jailbreak, PII leak, toxicity/bias — each with rationale + pass condition
- [ ] **4.2 Payload sets**
  - · `adversarial/payloads/` JSON per category; each case has input + expected safe behavior
- [ ] **4.3 Automated runner with a graded severity scale**
  - · `adversarial/run.py` executes payloads against the agent
  - · Result is **not binary** — use a 3-level rubric: **safe / partial-leak / breach** (real red-team work has partial outcomes; this reads like a security tester wrote it)
  - · PII-leak check: does it leak corpus/system-prompt content? Injection: does it stay grounded/refuse?
- [ ] **4.4 LLM-as-judge (Claude) where needed**
  - · For toxicity/bias, a Claude judge prompt; document its known failure modes (validated in Phase 7)
- [ ] **4.5 Wire into CI**
  - · Add replayed adversarial run to fast CI; live variant to the scheduled job

🎯 **Exit:** `make redteam` produces a graded report; catalog reads like a security tester wrote it.

---

## Phase 5 — Suite 3: Agent Reliability (Project 3 — the rare skill)
*Goal: test the graph, not just the final text.*

- [ ] **5.1 Tool-call correctness**
  - · Assert right tool + right args for representative inputs — **primary source = LangGraph's in-memory intermediate steps** (works offline, no key); LangSmith trace is the *also-visualized* bonus, never the only path
- [ ] **5.2 Termination & loop safety**
  - · Max-step guard; test that the agent halts and doesn't loop infinitely
- [ ] **5.3 State integrity**
  - · Multi-turn case: assert state carries correctly across steps
- [ ] **5.4 Failure recovery**
  - · Inject a tool failure (mock raises); assert graceful handling, not a crash
- [ ] **5.5 Trace-based assertions (graceful degradation by design)**
  - · Helper asserts on graph structure from **in-memory steps as the default path** so `make agent-tests` runs with **no `LANGSMITH_API_KEY`** (protects the "clone → run in 5 min, no keys" promise)
  - · *If* a LangSmith key is present, a second helper pulls the remote run and asserts on it too — documented as the "real-platform" pattern. Missing key → the LangSmith-specific test **skips with a clear message**, it does not fail
  - · Document both patterns in README

🎯 **Exit:** `make agent-tests` green **with and without** a LangSmith key; README shows you test agentic behavior, not just outputs.

---

## Phase 6 — Meta-Eval: Challenge the Judge (the sophistication signal)
*Goal: prove we don't trust the evaluator blindly. The judge is itself a model that can be wrong — so we measure it.*

- [ ] **6.1 Hand-labeled gold set**
  - · `meta_eval/gold.jsonl`: ~15–25 (answer, context) cases **you** have labeled with the correct verdict (grounded/not, safe/breach, etc.)
  - · Include deliberately tricky cases where a naive judge would slip
- [ ] **6.2 Judge agreement measurement + threshold calibration**
  - · Run the Claude judge over the gold set; compute agreement with your labels (accuracy + confusion matrix; Cohen's κ if you want to flex)
  - · **Calibrate the metric thresholds here** (closes the loop with the Threshold-strategy box): pick each pass line to best separate human-pass from human-fail, record the resulting number + this run's ID into `thresholds.yaml`. This is what makes the thresholds defensible rather than arbitrary.
- [ ] **6.3 Document judge failure modes**
  - · Where does the judge disagree with you, and why? Write it up honestly — this is the part that separates "I used an eval framework" from "I understand the evaluator can be wrong"
  - · The measured disagreement rate also sizes the **threshold margin** (Threshold-strategy box, point 3)
- [ ] **6.4 Judge-drift check (the judge is a moving model too)**
  - · The generator-drift problem applies to Claude as well. Re-running this gold set on a schedule (alongside the live tier) **is** our judge-drift detector: if judge-vs-human agreement falls, the evaluator itself has drifted and thresholds may need re-calibration. State this explicitly in the README.
- [ ] **6.5 (Optional stretch) Cross-judge check**
  - · Spot-check a few cases with a second judge model; note where judges disagree with each other

🎯 **Exit:** `make meta-eval` reports judge-vs-human agreement, writes calibrated thresholds, and is re-runnable as a judge-drift check; README has an honest "how good is the judge?" section.

---

## Phase 7 — Narrative, Polish & Ship
*Goal: the README that makes a recruiter stop scrolling.*

- [ ] **7.1 Cost/quota budget doc**
  - · `docs/COST.md`: requests-per-live-run math = (#functional × N samples) + (#adversarial) + (#agent) + (#meta-eval), each ×(RAG call + judge call)
  - · Compare against Gemini free-tier RPD/RPM and Claude limits; define behavior at 429 (backoff/retry, then mark-skipped rather than fail)
  - · **You set the final per-run quota cap here once the suites exist**
- [ ] **7.2 Metrics-over-time surface**
  - · Append each live run's scores to a committed `evals/history/` (CSV/JSON); render a simple trend so drift is *visible*, not just pass/fail
- [ ] **7.3 README narrative arc**
  - · (1) Why testing AI differs from traditional assertions · (2) the suites as one strategy · (3) the record/replay-vs-live CI philosophy · (4) how thresholds are calibrated & defended (Threshold-strategy box) · (5) a caught-regression walkthrough · (6) the stochasticity/determinism finding (`temp=0 ≠ determinism`) · (7) "how good is the judge" (meta-eval)
- [ ] **7.4 Architecture diagram**
  - · One diagram: app under test (Gemini) ← suites ← Claude judge ← two-tier CI
- [ ] **7.5 Quickstart**
  - · Clone → install → `make eval-ci` in under 5 minutes, no keys required
- [ ] **7.6 Write-up post (draft early!)**
  - · Draft "How I built CI that catches LLM hallucinations" **right after Phase 3** — writing the explanation exposes design holes (esp. the record/replay framing) while they're cheap to fix; finalize here
- [ ] **7.7 Final pass**
  - · Every `make` target works from a clean clone; remove dead code; tag a `v1.0` release

🎯 **Exit:** a stranger can understand the value in 30 seconds and run it in 5 minutes.

---

## Dependency order (do not reorder)
```
Phase 0 → 1 → 2 → 3  ← (Project 1 complete & shippable here)
                  ↘ 4  (adversarial, additive)
                  ↘ 5  (agent reliability, additive)
                  ↘ 6  (meta-eval — can start once any LLM-judge exists, i.e. after Phase 2)
                       → 7 (ship)
```
**Critical rule:** Phases 4, 5, and 6 are *additive layers*. If you stall, a finished Phase 0–3 is already a strong standalone portfolio piece. Never start a new suite with the previous one's CI red.

**Threshold-calibration caveat (resolve the one ordering tension):** Phase 2 *uses* thresholds but Phase 6 *calibrates* them. So Phase 2 ships with **provisional, hand-reasoned thresholds + a TODO**, and Phase 6 replaces them with gold-set-calibrated values. If you ship at Phase 3 (without Phase 6), say so honestly in the README — "thresholds are provisional pending meta-eval calibration." Do **not** claim calibrated thresholds before Phase 6 exists. This keeps the methodology claim truthful at every shippable checkpoint.

**Draft-early note:** begin the Phase 7.6 write-up right after Phase 3 — it's a design-review tool, not just marketing.
