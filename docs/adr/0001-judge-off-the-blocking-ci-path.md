# The LLM judge never gates a merge

The Claude judge is both non-deterministic and paid, so gating merges on it would make CI
red for reasons unrelated to the change under review, and would spend money on every push.
We split CI into three tiers instead: the **fast** tier (`ci.yml`) runs on every push with no
API keys, replaying committed recordings, and is the only blocking check; the **judged** tier
(`judged-eval.yml`) runs a genuinely live judge (`JUDGE_LIVE=1`, bypassing the verdict cache)
but only on `workflow_dispatch` or the `judged-eval` label, and never as a required check; the
**live drift** tier (`live-eval.yml`) runs weekly on cron against real Gemini and Claude.

Each tier answers a different question — "did my code regress?", "does the metric logic hold
against a live judge?", "did the model drift?" — and keeping them separate is what stops a
flaky judge from being confused with a real regression.

## Consequences

A judge regression is caught within a week (cron) or on demand (label), not at merge time.
This is deliberate: we accept slower detection in exchange for a blocking check that is
deterministic, free, and trustworthy. Do not "fix" the judged tier by making it a required
check — that reintroduces exactly the coupling this decision removes.
