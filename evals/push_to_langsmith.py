"""Push the eval dataset to LangSmith as a named dataset (plan 2.1).

This makes the eval set visible in the LangSmith UI — demonstrating real eval-tooling
use, not just local files. Idempotent: if the named dataset already exists, examples are
added/updated rather than erroring.

    python -m evals.push_to_langsmith

Requires LANGSMITH_API_KEY. No-op with a clear message if tracing isn't configured.
"""

from __future__ import annotations

from evals.dataset import load_cases
from shared import config

DATASET_NAME = "llm-test-harness-functional"


def main() -> None:
    if not config.langsmith_enabled():
        print(
            "LangSmith not configured (need LANGSMITH_API_KEY + LANGSMITH_TRACING=true). "
            "Skipping dataset push — this is optional."
        )
        return

    from langsmith import Client

    client = Client()
    cases = load_cases()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset {DATASET_NAME!r} exists; refreshing examples.")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Hand-authored functional eval cases for the Meridian RAG harness.",
        )
        print(f"Created dataset {DATASET_NAME!r}.")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": c.question} for c in cases],
        outputs=[
            {
                "reference_answer": c.reference_answer,
                "expected_sources": c.expected_sources,
                "type": c.type,
            }
            for c in cases
        ],
        metadata=[{"id": c.id, "type": c.type} for c in cases],
    )
    print(f"Pushed {len(cases)} examples to LangSmith dataset {DATASET_NAME!r}.")


if __name__ == "__main__":
    main()
