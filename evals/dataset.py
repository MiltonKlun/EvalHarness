"""Load the versioned eval dataset and the (provisional) thresholds.

Kept tiny and dependency-light so the test runner can parametrize over cases without
importing the heavy LLM stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_EVALS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVALS_DIR.parent
DATASET_PATH = _EVALS_DIR / "dataset.jsonl"
THRESHOLDS_PATH = _REPO_ROOT / "thresholds.yaml"


@dataclass(frozen=True)
class Case:
    """One eval case (a row of dataset.jsonl)."""

    id: str
    type: str  # "answerable" | "multihop" | "unanswerable"
    question: str
    reference_answer: str
    expected_sources: list[str]
    must_contain: list[str] = field(default_factory=list)
    hop_facts: list[str] = field(default_factory=list)
    abstain: bool = False

    @property
    def is_unanswerable(self) -> bool:
        return self.type == "unanswerable" or self.abstain


def load_cases() -> list[Case]:
    """Parse dataset.jsonl into Case objects."""
    cases: list[Case] = []
    with DATASET_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            cases.append(
                Case(
                    id=row["id"],
                    type=row["type"],
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    expected_sources=row.get("expected_sources", []),
                    must_contain=row.get("must_contain", []),
                    hop_facts=row.get("hop_facts", []),
                    abstain=row.get("abstain", False),
                )
            )
    return cases


def load_thresholds() -> dict[str, Any]:
    """Load thresholds.yaml (provisional until Phase 6 calibration)."""
    return yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))
