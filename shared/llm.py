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

import time
from typing import Any

from shared import cache, config

# Transient generator failures seen live: 503 UNAVAILABLE (high demand) and short-lived
# 429s. We retry these with backoff so a recording run survives a blip. A *sustained*
# quota 429 (daily cap) still surfaces after the retries — that's a real limit, not a blip.
_TRANSIENT_MARKERS = ("503", "unavailable", "overloaded", "deadline", "timeout")
_MAX_RETRIES = 3


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _with_retries(fn):
    """Call ``fn`` with backoff on transient errors; re-raise on the final attempt."""
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classify, then re-raise non-transient
            last = exc
            if not _is_transient(exc) or attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)  # 1s, 2s
    raise last  # pragma: no cover


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
        return _with_retries(lambda: model.invoke(prompt).content)

    return cache.cached_call(provider_model, prompt, params, _compute)
