"""Provider-agnostic LLM access — every model call in the project goes through here.

Built on LangChain's ``init_chat_model("<provider>:<model>")`` so swapping a provider
is a one-line config change (see ``shared.config``). Two named accessors encode the
independence property the build plan requires:

    generator() -> Gemini   (the system under test)
    judge()     -> Claude   (the evaluator — a different model family)

``complete()`` routes a single-prompt call through the record/replay cache, so the
same code path is reproducible offline (replay) and real-but-recorded live.
"""

from __future__ import annotations

from typing import Any

from shared import cache, config


def _init(provider_model: str, **kwargs: Any):
    """Lazily import + construct a LangChain chat model for ``provider_model``.

    Import is deferred so that offline/cached runs (which never construct a live
    client) don't pay the heavy LangChain import at module load.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(provider_model, **kwargs)


def generator(**kwargs: Any):
    """The Gemini model under test. Requires GOOGLE_API_KEY at call time."""
    config.require("GOOGLE_API_KEY")
    return _init(
        config.GENERATOR_MODEL,
        temperature=config.GENERATOR_TEMPERATURE,
        **kwargs,
    )


def judge(**kwargs: Any):
    """The Claude evaluator. Requires ANTHROPIC_API_KEY at call time."""
    config.require("ANTHROPIC_API_KEY")
    return _init(config.JUDGE_MODEL, **kwargs)


def complete(provider_model: str, prompt: str, **params: Any) -> str:
    """Single-prompt completion routed through the record/replay cache.

    In replay mode (default) this returns a recorded response and never touches the
    network. In live mode it calls the real model and records the result.
    """

    def _compute() -> str:
        model = _init(provider_model, **params)
        return model.invoke(prompt).content

    return cache.cached_call(provider_model, prompt, params, _compute)
