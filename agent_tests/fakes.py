"""Scripted fakes for keyless agent-graph testing.

``ScriptedModel`` plays back a fixed sequence of AIMessages (each either a tool call or a
final answer), so the reliability tests can drive the real LangGraph graph deterministically
with NO API key — exercising routing, the loop guard, state, and failure recovery.

This is the standard way to test agent *mechanics*: stub the model, keep the graph real.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.tools import tool


class ScriptedModel:
    """A minimal stand-in for a chat model that returns scripted AIMessages.

    The graph calls ``.bind_tools(...)`` then ``.invoke(messages)`` once per agent step;
    this returns the next scripted message each time it's invoked.
    """

    def __init__(self, script: list[AIMessage]):
        self._script = list(script)
        self._i = 0
        self.invocations: list[list] = []  # record inputs for assertions

    def bind_tools(self, _tools):
        return self  # tools don't affect a scripted model

    def invoke(self, messages):
        self.invocations.append(messages)
        if self._i >= len(self._script):
            # Past the script: emit a plain final answer so the graph always terminates.
            return AIMessage(content="(scripted model exhausted)")
        msg = self._script[self._i]
        self._i += 1
        return msg


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Build an AIMessage that requests a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final_message(text: str) -> AIMessage:
    """Build a plain final-answer AIMessage (no tool calls)."""
    return AIMessage(content=text)


# --- Fake tools for the graph (so we don't hit the real retriever/embeddings) ----------


@tool
def fake_retrieve(query: str) -> str:
    """Fake corpus retrieval — returns fixed text, no embeddings/key needed."""
    return f"FAKE CONTEXT for: {query}"


@tool
def boom(query: str) -> str:
    """A tool that always fails — used to test failure recovery (plan 5.4)."""
    raise RuntimeError("simulated tool failure")


FAKE_TOOLS = [fake_retrieve, boom]
