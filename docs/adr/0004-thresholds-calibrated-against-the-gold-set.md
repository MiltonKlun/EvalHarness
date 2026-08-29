# Pass/fail thresholds are calibrated against a hand-labeled gold set, not guessed

Metric cutoffs like "faithfulness must exceed 0.7" are usually picked by feel. Ours are
derived: `make meta-eval` finds the cutoff that best separates a set of 20 hand-labeled gold
cases, then subtracts a 0.05 margin to stay off the noisy boundary. The result is committed to
`thresholds.yaml` along with the calibration run id (`calibration_run: meta-eval-2026-06-29`)
and a `calibrated: true` marker, so a threshold's provenance is always recoverable.

The same gold set measures the judge itself — 80% accuracy, Cohen's kappa 0.60, with a
documented lenient bias (JUDGE-001) — which is why the thresholds carry the caveat that
faithfulness scores are an **upper bound** on true groundedness.

## Consequences

Thresholds are not free parameters: changing one by hand silently breaks the link between the
number and the evidence for it. Re-run the meta-eval to re-derive them instead. The gold set
is hand-authored, so the system cannot teach to its own test — but it is also small (20
cases), which bounds how finely the cutoffs can be resolved.
