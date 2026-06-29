"""Agent reliability suite (plan 5.1-5.4) — tests the GRAPH, not just final text.

All keyless: a ScriptedModel drives the real LangGraph graph with fixed tool-call
sequences, and we assert on the in-memory intermediate steps. No Gemini/Anthropic key, no
network — so `make agent-tests` runs on a fresh clone.
"""

from __future__ import annotations

from agent_tests.fakes import (
    FAKE_TOOLS,
    ScriptedModel,
    final_message,
    tool_call_message,
)
from agent_tests.trace import called_tools, tool_calls, tool_results
from app import agent
from app.agent import MAX_STEPS, invoke_agent

# --- 5.1 Tool-call correctness -----------------------------------------------------


def test_calls_right_tool_with_right_args():
    """The agent invokes the expected tool with the expected args, then answers."""
    script = [
        tool_call_message("fake_retrieve", {"query": "wind tolerance"}),
        final_message("The Kestrel-2 tolerates 22 m/s. (source: fake)"),
    ]
    state = invoke_agent(
        "which drone for high wind?", model=ScriptedModel(script), tools=FAKE_TOOLS
    )

    calls = tool_calls(state)
    assert len(calls) == 1
    assert calls[0].name == "fake_retrieve"
    assert calls[0].args == {"query": "wind tolerance"}
    # The tool actually ran and its result is in the trace.
    assert any("FAKE CONTEXT" in r for r in tool_results(state))


def test_multi_tool_sequence_order():
    """Two tool calls in sequence are recorded in order."""
    script = [
        tool_call_message("fake_retrieve", {"query": "a"}, call_id="c1"),
        tool_call_message("fake_retrieve", {"query": "b"}, call_id="c2"),
        final_message("done"),
    ]
    state = invoke_agent("q", model=ScriptedModel(script), tools=FAKE_TOOLS)
    assert called_tools(state) == ["fake_retrieve", "fake_retrieve"]


# --- 5.2 Termination & loop safety -------------------------------------------------


def test_loop_guard_halts_infinite_tool_calling():
    """A model that NEVER stops calling tools must still terminate at MAX_STEPS."""
    # Script longer than MAX_STEPS: every step requests another tool call.
    forever = [tool_call_message("fake_retrieve", {"query": f"q{i}"}) for i in range(MAX_STEPS + 5)]
    state = invoke_agent("q", model=ScriptedModel(forever), tools=FAKE_TOOLS)

    # It halted (didn't hang) and respected the guard.
    assert state["steps"] <= MAX_STEPS
    assert len(called_tools(state)) < len(forever)  # stopped before exhausting the script


def test_normal_path_terminates_without_hitting_guard():
    script = [tool_call_message("fake_retrieve", {"query": "x"}), final_message("answer")]
    state = invoke_agent("q", model=ScriptedModel(script), tools=FAKE_TOOLS)
    assert state["steps"] < MAX_STEPS


# --- 5.3 State integrity -----------------------------------------------------------


def test_state_accumulates_messages_and_step_count():
    """State carries correctly across steps: messages grow, step counter advances."""
    script = [tool_call_message("fake_retrieve", {"query": "x"}), final_message("final")]
    state = invoke_agent("the question", model=ScriptedModel(script), tools=FAKE_TOOLS)

    # The original human question is still present at the head of the accumulated history.
    from langchain_core.messages import HumanMessage

    assert any(
        isinstance(m, HumanMessage) and "the question" in m.content for m in state["messages"]
    )
    # steps advanced once per agent turn (2 turns: tool-call turn + final turn).
    assert state["steps"] == 2
    # Final message is the answer text.
    assert agent._text_of(state["messages"][-1]) == "final"


# --- 5.4 Failure recovery ----------------------------------------------------------


def test_tool_failure_is_handled_not_crashed():
    """A raising tool must not crash the graph; the failure surfaces as a tool result."""
    script = [
        tool_call_message("boom", {"query": "trigger"}),  # this tool raises
        final_message("I hit an error but recovered."),
    ]
    # ToolNode catches the exception and returns it as a ToolMessage by default.
    state = invoke_agent("q", model=ScriptedModel(script), tools=FAKE_TOOLS)

    results = tool_results(state)
    assert any("simulated tool failure" in r or "error" in r.lower() for r in results)
    # The graph still produced a final answer rather than throwing.
    assert agent._text_of(state["messages"][-1]) == "I hit an error but recovered."
