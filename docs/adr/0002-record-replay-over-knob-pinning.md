# Reproducibility comes from committed recordings, not from pinning decode knobs

The obvious way to make an LLM test reproducible is to pin every knob the API offers —
`temperature=0`, `top_p`, `top_k`, `seed` — and assume identical output. We measured this
instead of assuming it, and it does not hold: within a single session the pinned pipeline
produced identical output, but across sessions (2026-07-04, every knob still pinned) at least
3 of 5 re-sampled cases produced *different* output. Google documents `seed` as best-effort.

So the source of reproducibility here is the **record/replay cache**: pay for each stochastic
call once, commit the recording, and re-run the free deterministic metric *code* over it on
every CI run. A cache miss in offline/replay mode raises `CacheMiss` — a hard failure by
design, never a silent live call.

## Consequences

Recordings are committed artifacts and must be regenerated deliberately when prompts or
models change; a stale recording is a real failure mode. In exchange the whole suite runs
offline, keyless, and identically on every machine. Pinned decode knobs are still set, but
they are a nice-to-have, not the mechanism — do not remove the cache on the theory that
pinning is sufficient. Evidence: `docs/determinism_run.txt`.
