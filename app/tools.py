"""Tools the agent can call.

Two tools, deliberately: the retrieval tool (the real capability) and one *external*
tool stub. The stub is intentionally trivial — Phase 5 tests the agent-reliability
*technique* (right tool, right args, loop safety, failure recovery), not a sophisticated
agent. Keeping the external tool a stub is a deliberate scope choice (plan Phase 5 note).
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.retriever import retrieve


@tool
def retrieve_corpus(query: str) -> str:
    """Search the Meridian Robotics document corpus and return the most relevant passages.

    Use this for any question about Meridian's company, products, or safety policy.
    """
    chunks = retrieve(query)
    if not chunks:
        return "No relevant passages found."
    return "\n\n".join(f"(source: {c.source})\n{c.text}" for c in chunks)


@tool
def get_wind_conditions(site: str) -> str:
    """Get the current wind speed (m/s) at an offshore wind-farm site.

    This is a stub external tool: it returns a fixed reading rather than calling a real
    weather API, so the harness stays self-contained and deterministic.
    """
    # Fixed stub reading — real implementation would call a weather service.
    return f"Current wind speed at {site}: 18 m/s, gusting to 21 m/s."


TOOLS = [retrieve_corpus, get_wind_conditions]
