"""Offline structural tests for the agent + RAG wiring — no API keys, no network.

These don't exercise the live model (that needs GOOGLE_API_KEY and is the Phase 1
exit check). They verify the pieces that must hold regardless of the model:
  - the tool surface is what we expect,
  - the prompt enforces grounding + abstention,
  - decode modes are well-formed (max_pinned actually pins every knob),
  - the agent graph compiles and has the loop-safety guard.
"""

from __future__ import annotations

from app import rag
from app.tools import TOOLS


def test_tool_surface():
    names = {t.name for t in TOOLS}
    assert names == {"retrieve_corpus", "get_wind_conditions"}


def test_prompt_enforces_grounding_and_abstention():
    p = rag.SYSTEM_PROMPT.lower()
    assert "only" in p  # grounded-only instruction
    assert "i don't know based on the provided documents" in p  # exact abstention string


def test_decode_modes_pin_what_they_claim():
    # near_det: temperature only.
    assert rag.DECODE_MODES["near_det"] == {"temperature": 0}
    # max_pinned: every knob the API exposes, including the (best-effort) seed.
    mp = rag.DECODE_MODES["max_pinned"]
    assert mp["temperature"] == 0
    assert "top_p" in mp and "top_k" in mp and "seed" in mp


def test_unknown_decode_mode_rejected():
    import pytest

    with pytest.raises(ValueError):
        rag.answer("anything", mode="nonsense")


def test_agent_graph_compiles_and_has_guard():
    # Build the graph with a stubbed generator so no API key is needed.
    from unittest.mock import patch

    from app import agent

    class _FakeModel:
        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):  # pragma: no cover - not called here
            raise AssertionError("model should not be invoked in a structural test")

    with patch("app.agent.llm.generator", return_value=_FakeModel()):
        compiled = agent._build_graph()

    assert compiled is not None
    assert agent.MAX_STEPS >= 1  # loop-safety guard exists (tested for real in Phase 5)
