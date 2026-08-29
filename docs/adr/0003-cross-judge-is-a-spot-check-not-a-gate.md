# The cross-judge check is a spot-check, never a second gate

`meta_eval/cross_judge.py` scores the gold set with a second judge (judge-B, `openai:gpt-4o`
— a third model family, distinct from both the Gemini generator and the Claude judge-A) and
reports where the two judges agree, disagree, and how each sits against the human labels.

It is deliberately **read-only**: it never blocks CI, never enters the metric path, and never
touches the pinned `JUDGE_MODEL` or the calibrated thresholds. Its purpose is illustrative —
"would a different family's judge reach the same verdicts?" — for the meta-eval write-up.

The alternative, ensembling two judges into the metric path, was rejected: it would double the
cost and latency of every judged run, and there is no principled way to break a tie between
two fallible judges without a third opinion. One measured judge with a documented bias
(JUDGE-001) is more honest than two averaged judges with an undocumented one.

## Consequences

Judge-B's scores are recorded once and committed (`meta_eval/scores_openai.json`), so the
spot-check replays keyless like everything else. Adding a second judge to the metric path or
promoting this to a required check would invalidate the calibration in `thresholds.yaml`,
which is derived from judge-A alone.
