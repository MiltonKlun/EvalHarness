"""LangGraph agent: a tool-calling graph over the corpus + one external tool.

The graph is the *agentic* surface the Phase 5 reliability suite tests (tool-call
correctness, loop/termination safety, state integrity, failure recovery). It is kept
deliberately small — the tests are the star.

Shape:
    agent ──(tool calls?)──> tools ──> agent ──> ... ──> END
A max-step guard bounds the loop so the agent always terminates (plan 5.2).

CLI:
    python -m app.agent "Which drone can fly in 20 m/s wind?"
"""

from __future__ import annotations

import sys
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.tools import TOOLS
from shared import config, llm

# Max agent<->tools round trips before we force a stop. Loop-safety guard (plan 5.2).
MAX_STEPS = 6

AGENT_SYSTEM_PROMPT = """You are an assistant for Meridian Robotics field operators.

Use the retrieve_corpus tool to answer questions about the company, its products, or its
safety policy. Use get_wind_conditions when a question depends on current wind at a site.

Answer ONLY from tool results. If the documents do not contain the answer, say:
"I don't know based on the provided documents." Do not invent details. Cite your sources.
"""


class AgentState(TypedDict):
    """Graph state: the running message list plus a step counter for the loop guard."""

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


def build_graph(model=None, tools=None):
    """Construct and compile the agent graph.

    ``model`` and ``tools`` are injectable so the Phase 5 reliability tests can drive the
    graph with a scripted fake model + fake tools — exercising routing, the loop guard,
    state, and failure recovery with NO API key. In production both default to the real
    Gemini generator and the real tool set.
    """
    tools = tools if tools is not None else TOOLS
    if model is None:
        model = llm.generator()
    model = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        # On the first turn, prepend the system prompt.
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]
        response = model.invoke(messages)
        return {"messages": [response], "steps": state["steps"] + 1}

    def route(state: AgentState) -> str:
        """Continue to tools if the last message has tool calls and we're under budget."""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls and state["steps"] < MAX_STEPS:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    # handle_tool_errors=True turns a raising tool into a ToolMessage the agent can recover
    # from, instead of crashing the graph (failure-recovery property, plan 5.4 / VULN-driven).
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# Backwards-compatible private alias (used by existing structural test).
_build_graph = build_graph


def invoke_agent(question: str, model=None, tools=None) -> AgentState:
    """Run the agent and return the FULL final state (messages + steps).

    Returning the whole state — not just the answer text — is what lets the reliability
    suite assert on intermediate steps (which tools were called, with what args, how many
    steps the loop took). This is the "test the graph, not just the final text" surface.
    """
    app = build_graph(model=model, tools=tools)
    return app.invoke({"messages": [HumanMessage(content=question)], "steps": 0})


def run(question: str) -> str:
    """Run the agent end-to-end and return the final answer text."""
    config.langsmith_enabled()  # enable tracing if a key is present; harmless otherwise
    final = invoke_agent(question)
    return _text_of(final["messages"][-1])


def _text_of(message: AnyMessage) -> str:
    """Extract plain answer text from a message.

    Newer Gemini models return ``content`` as a list of blocks (text + a thinking
    signature) rather than a bare string; normalise both shapes to the answer text.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app.agent "your question"')
        raise SystemExit(2)
    question = " ".join(sys.argv[1:])
    print(run(question))


if __name__ == "__main__":
    main()
