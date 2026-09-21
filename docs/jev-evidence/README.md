# Jev evaluation — raw evidence

Raw outputs behind `docs/typesafe-jev-evaluation.md`. Committed so the numbers in that
document are auditable rather than asserted.

Measured 2026-09-21 against `jev-1.13.0`. 140 live calls, ~$0.0002 total.

| file | what it holds |
|---|---|
| `exp1_gold_tuned.json` | All 20 gold cases, tuned criteria. Per-case option, full probability distribution, confidence, latency. The 100% / κ 1.00 headline. |
| `exp2_criteria_ablation.json` | 3 arms × 20 cases: bare option names (90%), generic descriptions (90%), tuned (100%). The key result — the lift is the criteria, not the model. |
| `exp3_stability_3runs.json` | 3 identical repeats of the tuned arm. 0/20 option flips, 6/20 confidence changes. |

## Re-running

Needs `TYPESAFE_API_KEY` in `.env` and `typesafe-sdk` installed (`uv pip install typesafe-sdk`).
These are **live-only** scripts — deliberately NOT routed through `shared.cache`, because they
are one-off investigation tools, not part of the suite. Nothing in `make test` depends on them.

    .venv/Scripts/python.exe docs/jev-evidence/run_exp1.py
    .venv/Scripts/python.exe docs/jev-evidence/run_exp2_ablation.py
    .venv/Scripts/python.exe docs/jev-evidence/run_exp3_stability.py

Each writes its JSON next to itself. Results will not be byte-identical between runs:
the chosen option was stable across 4 observed runs, but confidence varies (see exp3).
