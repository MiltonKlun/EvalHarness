"""Trace-based assertions (plan 5.5) — in-memory default, LangSmith optional.

Demonstrates the project's graceful-degradation rule: the agent's structure is asserted
from LangGraph's in-memory steps (keyless, always works); the LangSmith remote-trace path
is a bonus that SKIPS cleanly when no key is configured — never the only way to test.
"""

from __future__ import annotations

import pytest

from agent_tests.fakes import FAKE_TOOLS, ScriptedModel, final_message, tool_call_message
from agent_tests.trace import called_tools
from app.agent import invoke_agent
from shared import config


def test_inmemory_trace_is_the_default_assertion_surface():
    """Primary path: assert tool structure from in-memory steps, no key needed."""
    script = [tool_call_message("fake_retrieve", {"query": "x"}), final_message("ok")]
    state = invoke_agent("q", model=ScriptedModel(script), tools=FAKE_TOOLS)
    assert called_tools(state) == ["fake_retrieve"]


@pytest.mark.skipif(
    not config.langsmith_enabled(),
    reason="LangSmith not configured — remote-trace assertions are a bonus, not required",
)
def test_langsmith_trace_path_available_when_configured():
    """Bonus path: when a LangSmith key is present, the remote-trace helper is importable
    and callable. We don't make a real remote round trip in CI; we assert the integration
    point exists so the 'real-platform' pattern is exercised when keys are available."""
    from agent_tests.trace import langsmith_tool_calls

    assert callable(langsmith_tool_calls)
