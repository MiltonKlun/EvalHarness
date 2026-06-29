"""Live tool-call correctness (plan 5.1) against the REAL Gemini agent — LIVE-only.

The scripted tests prove the graph *mechanics* keylessly; this proves the real model
actually picks the right tool for a representative input.

Unlike the RAG path, the agent graph does not (yet) route its model calls through the
record/replay cache — the agent invokes a bound chat model directly. So this test makes a
real Gemini call and is gated to LIVE_LLM=1: it SKIPS in the default keyless mode, keeping
`make agent-tests` green on a fresh clone, and runs in the live drift tier.
"""

from __future__ import annotations

import pytest

from agent_tests.trace import called_tools
from app.agent import invoke_agent
from shared import config


@pytest.mark.skipif(
    not config.LIVE_LLM,
    reason="agent path isn't cache-backed; live tool-choice check runs only with LIVE_LLM=1",
)
def test_real_agent_uses_retrieval_for_a_corpus_question():
    """A corpus question should drive the agent to call retrieve_corpus."""
    try:
        state = invoke_agent("How many employees does Meridian Robotics have?")
    except config.ConfigError as exc:
        pytest.skip(f"missing config for live run: {exc}")
    except Exception as exc:  # noqa: BLE001
        # A quota 429 / transient error shouldn't turn the live tier red — skip with a note.
        if any(m in str(exc).lower() for m in ("resource_exhausted", "429", "503", "unavailable")):
            pytest.skip(f"live API unavailable (quota/transient): {type(exc).__name__}")
        raise

    assert "retrieve_corpus" in called_tools(state), (
        f"expected retrieve_corpus, got {called_tools(state)}"
    )
