# Evaluating TypeSafe's Jev for EvalHarness

**Status: MEASURED, 2026-09-21; control experiment added 2026-09-22 (§8).** 140 live
`jev-1.13.0` calls over this repo's own gold set, total spend **$0.0002**. Every number below
is measured unless explicitly marked **[UNTESTED]**. Raw outputs: `docs/jev-evidence/`.

**Read §8 before acting on §1.** It found that judge-A's own metric already has the
"true but unsupported" bucket and discards it by default — which may make Jev unnecessary.

*(An earlier revision of this file was a design review written without an API key. It has been
superseded throughout; where a prediction turned out wrong, §7 says so.)*

Docs read: `llms.txt`, `models.md`, `model-jaggedness/jev-1.13.md`, `cookbooks/citation_check.md`,
plus the official agent skill (`typesafe@typesafe-ai` v0.5.7, installed).
Verified current as of 2026-09-20: **jev-1.13.0**, $0.042/M input tokens, output free,
64k context (32k state + longest question), 250k tok/s, 1,200 req/min.

---

## 1. The decisive experiment

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

### Result: Jev separates all four — but the prompt is doing much of the work

**Headline (tuned criteria, n=20): accuracy 100%, Cohen's κ 1.00, fp=0, fn=0.**
Judge-A (Claude) on the same set: 80%, κ 0.60, fp=4. All four known false positives landed in
`says_nothing`:

| case | Jev | confidence | distribution |
|---|---|---|---|
| `g_correct_world_fact_unsupported` | `says_nothing` | 0.76 | says_nothing 0.85 / supports 0.15 |
| `g_overconfident_inference` | `says_nothing` | 0.95 | says_nothing 0.97 / contradicts 0.03 |
| `g_invented_specific` | `says_nothing` | 0.53 | says_nothing 0.69 / contradicts 0.31 |
| `g_hedge_then_invent` | `says_nothing` | 1.00 | says_nothing 1.00 |

Mapping is a **code** decision, not a model output: `supports`→grounded,
`contradicts`/`says_nothing`→ungrounded. Scored with this repo's own `meta_eval/stats.py`.

**A perfect score is a reason for suspicion, not celebration.** My criteria text named the
JUDGE-001 failure mode almost explicitly ("TRUE IN THE REAL WORLD but simply absent", "even if
`answer` first acknowledges the information is missing"). That is arguably teaching to the
test, so I ablated it — 3 arms × 20 cases:

| arm | criteria | accuracy | errors |
|---|---|---|---|
| **A_bare** | option names only, no descriptions | **90%** | `g_correct_world_fact_unsupported` → supports; `g_hedge_then_invent` → supports |
| **B_generic** | cookbook-style, failure mode unmentioned | **90%** | `g_correct_world_fact_unsupported` → supports; `g_abstention_grounded` → says_nothing |
| **C_tuned** | names the failure mode (the headline) | **100%** | none |

**This is the most important result in the evaluation.** The 100% is *not* raw model
capability — it is capability **plus** a prompt written by someone who had already read
JUDGE-001. Unprompted, Jev reproduces Claude's exact lenient bias:
`g_correct_world_fact_unsupported` — the "Aberdeen is in the UK" case — fails in **both**
weaker arms. The honest claim is:

> The three-way Choice *gives you a place to put* "true but unsupported", and Jev will use it
> **if you describe that bucket explicitly**. The lift comes from the decomposition plus the
> criteria, not from the model spontaneously being stricter than Claude.

Arm B is also a warning about over-tuning in the other direction: it produced a *false
negative* (`g_abstention_grounded` → `says_nothing`), i.e. a correct abstention graded as
unsupported. Judge-A has zero false negatives. Criteria wording moves errors around, not just
away.

**Stability (3 identical repeats of the tuned arm, 60 calls):** accuracy 100% / 100% / 100%,
and **0/20 cases changed their chosen option**. But **6/20 cases returned a different
confidence** across runs — `g_invented_specific` ranged **0.48 → 0.63**, straddling 0.5. So the
*decision* is reproducible here while the *probability* is not, which matters directly for
constraint 4: a Jev threshold anywhere near 0.5 would sit on measurement noise. Any future Jev
cutoff needs the same margin discipline `thresholds.yaml` already applies to judge-A.

**Measured cost and latency:** ~638 input tokens per 20-case run ≈ **$0.000027/run**; all 140
calls in this investigation cost **~$0.0002** total. Mean latency **0.35 s** (min 0.29, max
0.93) — roughly an order of magnitude faster than a generative judge call.

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

**Cost (measured).** A 20-case gold run is ~638 input tokens ≈ **$0.000027**.
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

*Update 2026-09-22:* the Claude judge is now blocked too — the Anthropic credit balance is
exhausted (§8), which also failed the 2026-09-14 weekly live run. That is a **billing state,
not a rate limit**, and Jev only relieves it by *replacing* the judge, which §3 rules out until
measured. It is not an argument for Jev; restoring credit is.

---

## 3. Recommendation

**Jev does not enter the metric path — and on the measured evidence, the cheaper move is to
test the question shape on judge-A first.**

Three things follow from §1, in priority order:

1. **Try the fix without the vendor — and it may be a one-flag change.** The ablation shows
   most of the lift came from explicit "true-but-unsupported" criteria, not from Jev being
   natively stricter. §8 found that judge-A's metric *already has that bucket*: DeepEval's
   faithfulness verdicts are `yes` / `no` / `idk`, and its default scorer counts `idk` as
   faithful. `FaithfulnessMetric(penalize_ambiguous_claims=True)` counts it as unfaithful.
   Measuring that on the gold set is one `make meta-eval` run — no prompt edits, no new
   dependency. **Do this before adopting anything.** Blocked today only by Anthropic credit.
2. **If Jev is adopted anyway, it arrives as a spot-check** — exactly as judge-B did, routed
   through `shared.cache`, with its scores committed, never gating a merge. That follows from
   the repo's own standard (JUDGE-001, ADR-0003) and from TypeSafe's docs, which state plainly
   that typed output guarantees *shape, not truth*.
3. **A 100% on 20 cases is not a licence.** It is one small hand-built set, scored with criteria
   written by someone who had read the answer key. Treat it as a promising signal, not a result.

Nothing here is a special concession to a new vendor; it is the rule the project already
applies to itself.

**Independence — the argument holds, and it is stronger than for judge-B.** Generator is
Gemini, judge-A is Claude, judge-B is OpenAI. Jev is both a fourth family *and* a different
**kind** of evaluator: constrained typed judgment rather than generated text. Judges A and B
share an architecture and a failure mode (both are prompted generative LLMs, both can
rationalize). A System One model can fail, but it cannot fail in quite the same way.

**Measured, this argument is weaker than it looks.** The ablation showed Jev making *exactly*
judge-A's mistake on `g_correct_world_fact_unsupported` when the criteria didn't name the
failure mode. A different architecture did not buy a different failure mode for free — the
prompt did. Independence of *family* is real; independence of *failure mode* was not
demonstrated.

Be precise about what the distribution buys, though. Jev returns `probabilities` and a
`confidence` where a prompted judge returns a number it made up — but per TypeSafe's own
confidence page, that confidence reflects *distribution concentration, not correctness*.

**Measured, and the honest answer is "unfalsifiable on this data":** with 20/20 correct there
are **zero errors to correlate confidence against**. Mean confidence on correct answers was
0.943. The tempting reading — "low confidence flags the hard cases" — is *weakly* supported at
best: the two lowest-confidence cases (0.53, 0.76) were indeed two of the four JUDGE-001 cases,
which is suggestive. But a signal that never fired on a wrong answer has not been shown to
predict wrongness. Claiming calibration here would be exactly the kind of unearned number this
repo exists to avoid. A bigger or harder set would be needed. **[UNTESTED — by construction]**

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

## 5. Draft finding — JUDGE-003, ready to commit

Measured. Drafted in the existing FINDINGS.md style. The headline is deliberately the
*ablation*, not the 100% — the 100% alone would be a misleading advertisement.

> ## JUDGE-003 — A typed three-way judge separates "true but unsupported", but only when the
> criteria say so
>
> | field | value |
> |---|---|
> | **Type** | judge comparison (not an agent defect) |
> | **Discovered by** | `jev-1.13.0` (TypeSafe System One) over `meta_eval/gold.jsonl`, 2026-09-21 |
> | **Severity** | informational — spot-check, never a gate |
> | **Status** | **Measured.** 140 live calls, $0.0002 total |
>
> **Motivation.** JUDGE-001 documents a lenient bias: the Claude faithfulness judge passes
> claims true in the world but absent from context (4 false positives, each at score 1.00). A
> *binary* faithfulness score structurally cannot express "the context is silent on this" — it
> collapses *contradicted* and *unaddressed* into one axis. A three-way Choice
> (`supports` / `contradicts` / `says_nothing`) has a bucket for it.
>
> **Measurement.** 20 gold cases, one Choice per case, mapping `supports`→grounded and
> `contradicts`/`says_nothing`→ungrounded in code. With criteria that explicitly describe the
> failure mode: **accuracy 100%, Cohen's κ 1.00, fp=0, fn=0** (judge-A: 80%, κ 0.60, fp=4).
> All 4 known false positives landed in `says_nothing`.
>
> **The result that matters — an ablation, 3 arms × 20 cases.** With bare option names
> (no descriptions) accuracy is **90%**; with generic cookbook-style descriptions that do not
> mention the failure mode, also **90%**. In *both* weaker arms
> `g_correct_world_fact_unsupported` ("Aberdeen is in the UK") is graded `supports` — i.e.
> **Jev reproduces judge-A's exact lenient bias when unprompted.** The generic arm also
> introduced a *false negative* (`g_abstention_grounded`, a correct abstention graded
> unsupported), an error class judge-A does not make.
>
> **Conclusion.** The lift is **decomposition + explicit criteria**, not a model that is
> natively stricter. The three-way Choice gives you somewhere to put "true but unsupported";
> it does not, by itself, make a model notice it. This is a finding about *prompt structure*,
> and it applies to judge-A too: the same explicit boundary may be worth testing in the
> DeepEval faithfulness prompt, at zero vendor risk.
>
> **Reproducibility.** 3 identical repeats: accuracy 100%/100%/100% and **0/20 option flips**,
> but **6/20 cases returned a different confidence**, one ranging 0.48→0.63. The decision is
> stable; the probability is not. Any Jev threshold would need the same margin discipline
> `thresholds.yaml` applies to judge-A, and must never reuse judge-A's 0.5 cutoff — TypeSafe's
> own docs state there are no structural invariants across question types.
>
> **Confidence is not validated here.** With zero errors there is nothing to correlate
> confidence against. Mean confidence on correct answers was 0.943; the two lowest-confidence
> cases were JUDGE-001 cases, which is suggestive but not evidence.
>
> **Honest framing.** A spot-check on 20 cases, not a controlled cross-model study. It does
> not change the pinned judge or the calibrated thresholds, and it is not a licence to trust a
> vendor's calibration claim.

**A second finding remains available and is now more attractive, not less.** The jaggedness
page states Jev treats state as data and is not hardened against adversarial content. This
repo has a hand-authored injection/jailbreak corpus. Measuring whether red-team payloads move
Jev's own answers is a publishable result about a documented weakness on real adversarial
data. Given §1's result — that Jev's answer is quite sensitive to how the question is
worded — the hypothesis that hostile *state* also moves it is well motivated. **[UNTESTED]**

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
- **The brief's one blind spot: it asked whether Jev beats the judge, not whether the
  *question shape* beats the judge.** The ablation says most of the measured lift is the
  three-way decomposition plus explicit criteria. That is a portable idea, and testing it on
  judge-A costs nothing and adds no vendor. See §7.

---

## 7. Where the earlier design review was wrong

Recorded because this document previously made predictions without a key, and honesty about
the process is the same standard the repo applies to its numbers.

| prediction | outcome |
|---|---|
| "Whether Jev separates those 4 cases is unknown" | **Correct to hedge.** It does — all 4, with the tuned criteria. |
| Implied that separation would be a property of *the model* | **Wrong.** The ablation shows it is largely a property of *the criteria*. Unprompted, Jev makes judge-A's mistake. |
| "Confidence tracking accuracy is an empirical question" | **Correct, but unanswerable here** — 20/20 leaves no errors to correlate against. |
| Cost "rounds to zero" | **Confirmed**, and by a wider margin than estimated: $0.0002 for the entire investigation. |
| Quota argument unfavourable | **Unchanged.** The generator is still the binding constraint. |

**The finding the design review could not have produced** is the ablation. It is also the most
useful one for this repo, and it points somewhere cheaper than adopting Jev: if explicit
"true-but-unsupported" criteria are what fix the JUDGE-001 class, that boundary can be written
into the **existing** DeepEval faithfulness prompt and measured with `make meta-eval` — no
fourth vendor, no new dependency, no keyless-replay risk. **That experiment should be run
before any decision to adopt Jev**, because if it works it makes most of the Jev case moot.
**[UNTESTED]**

---

## 8. Control experiment: judge-A already has a `says_nothing` bucket (2026-09-22)

§7 proposed testing the "true-but-unsupported" boundary on the existing judge before adopting
Jev. Reading DeepEval's source to design that test changed the question.

**Verified from source (`deepeval` 4.0.7), no API calls needed:**

- The faithfulness verdict prompt asks whether each claim **contradicts** the context — not
  whether the context **supports** it. Its guidelines say: *"Only use 'no' if retrieval
  context DIRECTLY CONTRADICTS the claim"* and *"Use 'idk' for claims not backed up by
  context"*.
- `_calculate_score` (`faithfulness.py:375`) counts every verdict that is not `no` as
  faithful. **An `idk` scores exactly like a `yes`.**
- `FaithfulnessMetric(penalize_ambiguous_claims=True)` changes only that arithmetic (and the
  reason text): `idk` then counts as unfaithful. The claim, truth, and verdict prompts sent to
  the judge are unchanged.

So DeepEval's faithfulness is structurally a **contradiction detector**, and it already has
the three-way distinction Jev's citation-check Choice offers: `yes` ≈ `supports`,
`no` ≈ `contradicts`, `idk` ≈ `says_nothing`. The default setting throws the third bucket
away. That is a **plausible mechanism for JUDGE-001**: all four false positives scored exactly
1.00, which is what you would see if Claude had correctly answered `idk` and the scorer had
discarded it.

**Plausible is not shown.** A score of 1.00 is equally consistent with Claude answering `yes`
(genuinely lenient). Telling the two apart needs the per-claim verdicts for the gold set, and
those were never recorded — `meta_eval/scores.json` stores only final scores, and the committed
cache holds no gold-set judge calls. **[UNTESTED]**

**Blocked, not failed.** The live run (`JUDGE_LIVE=1`, cache bypassed so the baseline could
not be touched) stopped at the first call: `400 — Your credit balance is too low to access the
Anthropic API`. No calls were billed; the cache was 209 files before and after. The script is
committed as `docs/jev-evidence/run_exp4_idk_control.py`, ready to run once credit is restored.
It judges each case once and scores the *same* verdicts under both rules, so the comparison
isolates the scoring rule from judge stochasticity.

**What the committed functional-suite cache does show (keyless).** Across the 16 cached
faithfulness verdict calls: **17 `yes`, 1 `idk`, 0 `no`**. The one `idk` was *not* a
hallucination — Claude flagged a correct answer because it cited a source filename
(`02_products.md`) that is not part of the context text. Under
`penalize_ambiguous_claims=True` that correct answer drops from 1.00 to 0.50. So the flag has
a real cost: a **false negative on citations**, the same error class the Jev arm-B ablation
produced. Enabling it in the metric path would need the gold set to show the gain outweighs
that cost — and possibly a gold case for "cites a source filename".

*(Correction recorded for honesty: an intermediate tally in this session reported
"37 yes / 7 idk / 0 no". That mixed faithfulness with answer-relevancy calls, which DeepEval
also returns under a schema named `Verdicts`. The split figures above are the correct ones.)*

**What this does to the Jev case.** The three-way decomposition is *not* unique to Jev — judge-A
has it and scores it away by default. If `penalize_ambiguous_claims=True` recovers the four
JUDGE-001 cases, the remaining argument for Jev is speed (0.35 s) and price, neither of which is
a constraint here (§2). If it does *not* recover them — Claude answered `yes` — then JUDGE-001 is
genuinely a judgment problem, and the §1 result (Jev 100% with explicit criteria) becomes the
strongest evidence in this document.

**Next step, in order:**
1. Restore Anthropic credit.
2. `JUDGE_LIVE=1 .venv/Scripts/python.exe docs/jev-evidence/run_exp4_idk_control.py` (~60 Haiku
   calls, a few cents). Read the verdicts on the four JUDGE-001 cases first; they decide it.
3. Only then decide whether Jev earns a spot-check slot.
