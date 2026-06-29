"""Agreement statistics for the meta-eval — pure functions, no LLM, fully testable.

Given the human verdicts and the judge's predicted verdicts (both in {grounded,
ungrounded}), compute accuracy, a 2x2 confusion matrix, and Cohen's kappa. These are the
numbers that turn "I trust the judge" into "the judge agrees with me 90% of the time,
kappa 0.80 — here's where it slips."
"""

from __future__ import annotations

from dataclasses import dataclass

GROUNDED = "grounded"
UNGROUNDED = "ungrounded"


@dataclass(frozen=True)
class Confusion:
    """2x2 confusion matrix with 'grounded' as the positive class."""

    tp: int  # human grounded, judge grounded
    tn: int  # human ungrounded, judge ungrounded
    fp: int  # human ungrounded, judge grounded  (judge too lenient — missed a hallucination)
    fn: int  # human grounded, judge ungrounded  (judge too strict — flagged a good answer)

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


def confusion(human: list[str], judge: list[str]) -> Confusion:
    """Build the confusion matrix from aligned verdict lists."""
    if len(human) != len(judge):
        raise ValueError("human and judge verdict lists must be the same length")
    tp = tn = fp = fn = 0
    for h, j in zip(human, judge, strict=True):
        if h == GROUNDED and j == GROUNDED:
            tp += 1
        elif h == UNGROUNDED and j == UNGROUNDED:
            tn += 1
        elif h == UNGROUNDED and j == GROUNDED:
            fp += 1
        else:  # h grounded, j ungrounded
            fn += 1
    return Confusion(tp=tp, tn=tn, fp=fp, fn=fn)


def cohen_kappa(human: list[str], judge: list[str]) -> float:
    """Cohen's kappa: agreement corrected for chance. 1.0 = perfect, 0 = chance-level."""
    c = confusion(human, judge)
    n = c.total
    if n == 0:
        return 0.0
    po = c.accuracy  # observed agreement
    # Expected agreement by chance, from the marginals.
    human_grounded = (c.tp + c.fn) / n
    judge_grounded = (c.tp + c.fp) / n
    pe = human_grounded * judge_grounded + (1 - human_grounded) * (1 - judge_grounded)
    if pe == 1.0:
        return 1.0  # degenerate (all one class) — treat as full agreement
    return (po - pe) / (1 - pe)


def verdict_from_score(score: float, threshold: float) -> str:
    """grounded if score >= threshold, else ungrounded."""
    return GROUNDED if score >= threshold else UNGROUNDED


def calibrate_threshold(
    human: list[str], scores: list[float], grid: list[float] | None = None
) -> tuple[float, float]:
    """Pick the threshold that maximises agreement with the human labels.

    Returns (best_threshold, best_accuracy). Ties break toward the LOWER threshold
    (more lenient) only after we've maximised accuracy — callers add a safety margin
    separately (Threshold-strategy box, point 3).
    """
    grid = grid or [round(x / 100, 2) for x in range(0, 101, 5)]
    best_t, best_acc = grid[0], -1.0
    for t in grid:
        predicted = [verdict_from_score(s, t) for s in scores]
        acc = confusion(human, predicted).accuracy
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc
