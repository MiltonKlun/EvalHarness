# Evaluating TypeSafe's Jev for EvalHarness

**Status: design review only — NO live Jev calls were made.** `TYPESAFE_API_KEY` was not
available in this session, and the user elected to proceed without it. Every claim below is
either (a) sourced from the live TypeSafe docs, (b) sourced from this repo's code, or
(c) an untested prediction, explicitly marked **[UNTESTED]**. No number in this document was
measured against Jev. The decisive experiment has *not* been run.

Docs read: `llms.txt`, `models.md`, `model-jaggedness/jev-1.13.md`, `cookbooks/citation_check.md`,
plus the official agent skill (`typesafe@typesafe-ai` v0.5.7, installed).
Verified current as of 2026-09-20: **jev-1.13.0**, $0.042/M input tokens, output free,
64k context (32k state + longest question), 250k tok/s, 1,200 req/min.

---

## 1. The decisive experiment — designed, not run

**Hypothesis.** JUDGE-001 records a systematic lenient bias: the Claude faithfulness judge
passes claims that are *true in the world but absent from the context*. All 4 errors on the
gold set are false positives, each at the judge's max score of 1.00:

| case id | what the judge missed |
|---|---|
| `g_correct_world_fact_unsupported` | "Aberdeen is in the UK" — true, not in context |
| `g_overconfident_inference` | invented an extra job role |
| `g_invented_specific` | invented a price the context says is unpublished |
| `g_hedge_then_invent` | acknowledged the gap, then guessed anyway |

The citation-check cookbook attacks exactly this shape with one Choice over
`supports` / `contradicts` / `says_nothing`. **`says_nothing` is the bucket a binary
faithfulness score structurally cannot express** — it collapses "contradicted" and
"unaddressed" into one "not faithful" axis, and a lenient judge resolves the ambiguity
toward "faithful."

That is a genuine structural argument, and it is why this hypothesis ranks first. It is
*not* evidence. Whether Jev actually separates those 4 cases is unknown. **[UNTESTED]**

**The experiment to run when a key exists** (~20–40 calls, well under $0.01):

```python
from typesafe_sdk import TypeSafeClient, Choice

client = TypeSafeClient()   # reads TYPESAFE_API_KEY
for case in gold:           # meta_eval/gold.jsonl, 20 cases
    result = client.system_one(
        {"context": case["context"], "answer": case["answer"]},   # state: positional, JSON object
        {"relation": Choice(
            instructions="How does `context` relate to the claims made in `answer`?",
            criteria={
                "supports": "Every factual claim in `answer` is stated or directly implied by `context`.",
                "contradicts": "`context` states the opposite of a claim in `answer`.",
                "says_nothing": "`answer` makes a claim `context` does not address either way — "
                                "INCLUDING a claim that is true in the real world but absent here.",
            })},
        model="jev-1.13.0",   # pin the version; aliases move
    )
    ans = result.choices["relation"]
    ans.choice, ans.probabilities, ans.confidence   # option, full distribution, concentration
```

*API verified against the live SDK docs (`sdk/python/usage.md`, `sdk/python/api/clients/sync.md`,
`primitives/choice.md`) — `state` is positional and may be a JSON object; `criteria` values may be
descriptive strings; `model` is a valid per-call override; a Choice answer carries
`choice` / `probabilities` / `confidence`. Not executed.*

Report: per-case option + probability distribution + confidence; accuracy and Cohen's kappa
via the existing `meta_eval/stats.py`; and specifically whether the 4 known false positives
land in `says_nothing`. **A null result is publishable** — if Jev also passes them, that is a
finding about the failure class, not about Jev.

**Two design notes that matter more than they look.** First, the criteria text above spells
out "true in the real world but absent here" explicitly, because the jaggedness page says Jev
reads *literally* — it answers what you wrote, not what you meant. An implicit boundary would
likely fail. Second, `says_nothing` → `ungrounded` is a **mapping decision in code**, not a
model output; the gold set is binary and Jev's three-way answer must be projected onto it.

---

## 2. Ranked integration points

| # | Integration | Verdict | Tier | Survives keyless replay? |
|---|---|---|---|---|
| 1 | Judge-C on the gold set (meta-eval spot-check) | **Worth testing** | judged / live; replay in fast | Yes — mirror `cross_judge.py` |
| 2 | JUDGE-001 three-way citation check | **Worth testing** (same calls as #1) | as above | Yes |
| 3 | Adversarial: Jev-vs-injection measurement | **Worth testing as a FINDING** | live only | Yes, if recorded |
| 4 | Adversarial toxicity second opinion | **Weak** | — | Yes, but low value |
| 5 | `agent_tests/trace.py` decomposition | **No** | — | n/a |
| 6 | Anything in the metric path | **No** | — | n/a |

**Cost.** A 20-case gold run is roughly 20 × ~400 tokens ≈ 8k input tokens ≈ **$0.0003**.
Cost is not a constraint anywhere in this evaluation; it rounds to zero at every scale this
repo operates at.

**Keyless replay.** All of #1–#3 survive constraint 1 *only if* every Jev call routes through
`shared.cache.cached_call(provider_model, prompt, params, compute)`. That signature is a
clean fit — `provider_model="typesafe:jev-1.13.0"`, the serialized state+questions as
`prompt`, and the JSON response as the cached string. The `config.require("TYPESAFE_API_KEY")`
check must live *inside* the compute closure, exactly as `evals/judge.py:63` and
`meta_eval/cross_judge.py` already do, so replay never needs a key.

**Quota (question 5 in the brief, answered).** The bound on suite size in this repo is the
**Gemini generator** free tier — 20 requests/day, which truncated a determinism run on
2026-08-29. The Claude judge is not the bottleneck. Therefore **Jev's generous limits relieve
nothing**: it would add a fourth API dependency without loosening the constraint that actually
binds. This is an argument against breadth of adoption, and it is independent of Jev's quality.

---

## 3. Recommendation

**Jev does not enter the metric path. It arrives as a documented spot-check, exactly as
judge-B did — and only after the meta-eval measures it.**

This follows from the repo's own standard (JUDGE-001, ADR-0003) and from TypeSafe's docs,
which state plainly that typed output guarantees *shape, not truth*, and that calibration
must be validated on your own data. Nothing here is a special concession to a new vendor; it
is the rule the project already applies to itself.

**Independence — the argument holds, and it is stronger than for judge-B.** Generator is
Gemini, judge-A is Claude, judge-B is OpenAI. Jev is both a fourth family *and* a different
**kind** of evaluator: constrained typed judgment rather than generated text. Judges A and B
share an architecture and a failure mode (both are prompted generative LLMs, both can
rationalize). A System One model can fail, but it cannot fail in quite the same way. That is a
real independence gain and worth stating in the write-up — **if experiment 1 shows it
separates the cases.** If it doesn't, the architectural novelty is irrelevant.

Be precise about what the distribution buys, though. Jev returns `probabilities` and a
`confidence` where a prompted judge returns a number it made up — but per TypeSafe's own
confidence page, that confidence reflects *distribution concentration, not correctness*. It is
a better-typed uncertainty signal, not a validated one. Whether it tracks accuracy here is an
empirical question about our data, and it is unanswered. **[UNTESTED]**

**Thresholds.** Per constraint 4 and the jaggedness page's "no structural invariants": a Jev
Choice probability is **not comparable** to the faithfulness 0.5 cutoff. Any Jev threshold
needs its own calibration run id in `thresholds.yaml`, and only after it has earned a place —
which it has not.

---

## 4. Where Jev should NOT go

1. **The metric path / `thresholds.yaml`.** Not before the meta-eval measures it on our gold
   set. Constraint 2, ADR-0003, JUDGE-001.
2. **The blocking fast tier as a live call.** Replay only, always. Constraint 1.
3. **`agent_tests/trace.py`.** The current assertions — did it call the retriever, do the args
   match, does the result id match the call — are **deterministic, keyless, free, and
   correct**. TypeSafe's own first design rule is to keep deterministic work in code. The only
   genuinely semantic question here is "is this tool appropriate for this request," and with
   one retrieval tool over a fixed corpus that set is **nearly empty**. Adding a network call
   to a suite whose entire virtue is that it runs offline would be a net loss.
4. **Anything numeric or date-based.** The jaggedness page is explicit: counting, numeric
   proximity, and date ordering are unreliable. This repo's metrics are numeric aggregates —
   they belong in `stats.py`, which is pure, tested, and free.
5. **Text generation of any kind.** Jev does not generate. It cannot replace the Gemini
   generator or write report prose.
6. **As a replacement for judge-A or judge-B.** The point of a spot-check is that the pinned
   judge stays pinned. Swapping evaluators invalidates the calibration and the history trend.
7. **Grading adversarial output as the *only* judge.** See the caveat below — Jev is
   explicitly not hardened against adversarial content, and the red-team corpus is adversarial
   by construction.

---

## 5. Draft finding — JUDGE-003 (hold until measured)

**Do not commit this until experiment 1 runs.** Drafted in the existing FINDINGS.md style so
it is ready; the bracketed values are placeholders, not predictions.

> ## JUDGE-003 — A System One model as a third-family judge: [does / does not] separate the
> "true but unsupported" class
>
> | field | value |
> |---|---|
> | **Type** | judge comparison (not an agent defect) |
> | **Discovered by** | `meta_eval/jev_judge.py` over `meta_eval/gold.jsonl` |
> | **Severity** | informational — spot-check, never a gate |
> | **Status** | [measured / negative result] |
>
> **Motivation.** JUDGE-001 documents a lenient bias: Claude passes claims true in the world
> but absent from context. A binary faithfulness score cannot express "the context is silent
> on this" — it collapses *contradicted* and *unaddressed* into one axis. TypeSafe's Jev
> offers a three-way Choice (`supports` / `contradicts` / `says_nothing`) where that bucket is
> explicit.
>
> **Measurement.** [N] gold cases, model `jev-1.13.0`, one Choice per case.
> Accuracy [X]%, Cohen's κ [Y] (vs Claude's 80% / 0.60). Of the 4 known false positives,
> [n] landed in `says_nothing`.
>
> **Confidence calibration.** [Whether Jev's returned confidence tracked its accuracy on the
> gold set — a reliability curve, not a headline number. Caveat, from TypeSafe's own docs:
> confidence measures how *concentrated* the probability distribution is, NOT correctness. A
> confident wrong answer is entirely possible. Testing whether concentration happens to
> correlate with correctness on our data is exactly the kind of vendor claim this repo exists
> to check — and the raw `probabilities` may be more informative than the single number.]
>
> **Honest framing.** This is a spot-check, not a controlled cross-model study. It does not
> change the pinned judge or the calibrated thresholds. [If negative: the lenient-bias failure
> class survived a differently-shaped evaluator, which strengthens rather than weakens
> JUDGE-001 — the problem is the task, not the judge.]

**A second finding is available even if #1 fails.** The jaggedness page states Jev treats
state as data and is *not* hardened against adversarial content. This repo has a hand-authored
injection/jailbreak corpus. Measuring whether red-team payloads move Jev's own answers is a
publishable result about a documented weakness, tested on real adversarial data — and it is
the kind of finding this repo exists to produce. **[UNTESTED]**

---

## 6. What I'd challenge in the brief

- **"The strongest candidate of my three" — agreed, and for the stated reason.** The gold set,
  kappa statistics, calibrated thresholds, and the judge-B precedent mean Jev can be *measured*
  here rather than merely adopted. That is unusual and it is the whole argument.
- **Hypothesis 4 (`trace.py`) should be dropped, not merely approached skeptically.** The brief
  already suspects this; the code confirms it. One retrieval tool over a fixed corpus leaves
  essentially no semantic question, and the suite's keyless-offline property is worth more than
  any judgment Jev could add there.
- **Hypothesis 5's quota question has a definite answer, and it is unfavourable.** The binding
  constraint is the *generator* quota, not the judge. Jev relieves nothing.
- **Sequencing.** Experiment 1 gates everything. If Jev does not separate the JUDGE-001 cases,
  hypotheses 2 and 3 lose most of their value and the honest recommendation is "no, with a
  documented negative finding" — which is a perfectly good outcome and cheaper than the
  alternative.
