"""Trace-inspection helpers — assert on the agent's GRAPH, not just its final text.

Primary path (keyless, default): read LangGraph's in-memory intermediate steps straight
off the final state's message list. This works offline with no API key, which is what
keeps `make agent-tests` runnable on a fresh clone (plan 5.5).

Optional path (bonus): if a LangSmith key is present, the same structure can also be
pulled from the remote trace — see ``langsmith_tool_calls``. That path SKIPS cleanly when
no key is set; it is never the only way to test.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, ToolMessage


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


def tool_calls(state: dict) -> list[ToolCall]:
    """Every tool call the agent made, in order, from the in-memory messages."""
    calls: list[ToolCall] = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []) or []:
                calls.append(ToolCall(name=tc["name"], args=tc.get("args", {})))
    return calls


def tool_results(state: dict) -> list[str]:
    """Every tool result (ToolMessage content), in order."""
    return [m.content for m in state["messages"] if isinstance(m, ToolMessage)]


def called_tools(state: dict) -> list[str]:
    """Just the ordered list of tool names the agent invoked."""
    return [c.name for c in tool_calls(state)]


def langsmith_tool_calls(run_id: str) -> list[ToolCall]:
    """Bonus: reconstruct tool calls from a LangSmith run (the 'real-platform' pattern).

    Only usable when a LangSmith key is configured; callers should guard with
    ``shared.config.langsmith_enabled()`` and skip otherwise. Provided to demonstrate that
    the same assertions can run against a production tracing platform — never as the only
    path.
    """
    from langsmith import Client

    client = Client()
    run = client.read_run(run_id)
    calls: list[ToolCall] = []
    for child in client.list_runs(parent_run_id=run.id):
        if child.run_type == "tool":
            calls.append(ToolCall(name=child.name, args=child.inputs or {}))
    return calls
