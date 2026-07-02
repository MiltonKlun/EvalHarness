"""Record/replay cache for raw LLM responses — the CI-cost mechanism.

This is the heart of the two-tier CI strategy (see the build plan's "Two-tier CI"
box). It stores ONLY the raw LLM response text, keyed by a hash of everything that
determines that response (provider + model + params + prompt). The eval *metrics*
never live here — they re-run live on every CI run, on top of replayed inputs.

Modes (driven by ``config.LIVE_LLM`` and ``config.RECORD_MISSING``):
- LIVE_LLM truthy                    -> call ``compute()`` for real, record to disk.
- LIVE_LLM truthy + RECORD_MISSING   -> replay keys already on disk; call ``compute()``
                                        only for genuine misses (cheap re-record path).
- LIVE_LLM falsy                     -> replay from disk; a cache MISS is a hard error
                                        (CacheMiss), never a silent live call or pass.
  (RECORD_MISSING is a no-op when LIVE_LLM is falsy — replay never calls the model.)

That hard-fail-on-miss rule is what makes fast CI honest: if the app/prompt changes
such that a new request is issued, the key changes, the recording is absent, and CI
fails loudly instead of quietly reaching for the network.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shared import config


class CacheMiss(RuntimeError):
    """Raised in replay mode when no recording exists for a request key."""


def _key(provider_model: str, prompt: str, params: dict[str, Any]) -> str:
    """Stable hash of everything that determines the response.

    ``params`` is serialized with sorted keys so logically-identical requests hash
    identically regardless of dict ordering.
    """
    payload = json.dumps(
        {"pm": provider_model, "prompt": prompt, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def cached_call(
    provider_model: str,
    prompt: str,
    params: dict[str, Any],
    compute: Callable[[], str],
) -> str:
    """Return a recorded response, or (in live mode) compute and record one.

    Args:
        provider_model: e.g. ``"anthropic:claude-haiku-4-5"`` — part of the key.
        prompt: the fully-rendered prompt string — part of the key.
        params: decode params (temperature, etc.) — part of the key.
        compute: zero-arg callable that performs the real LLM call. Only invoked
            when ``config.LIVE_LLM`` is truthy.

    Raises:
        CacheMiss: in replay mode when no recording exists for this key.
    """
    key = _key(provider_model, prompt, params)
    path = _path(key)

    if config.LIVE_LLM:
        # RECORD_MISSING: in a live run, replay a key that is already recorded and only
        # call the model for genuine misses — the cheap re-record path (see config).
        if config.RECORD_MISSING and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            return record["response"]
        response = compute()
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "provider_model": provider_model,
            "prompt": prompt,
            "params": params,
            "response": response,
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return response

    if not path.exists():
        raise CacheMiss(
            f"No recorded response for key {key[:12]}… "
            f"(provider_model={provider_model!r}). "
            f"Run once with LIVE_LLM=1 to record, then commit evals/cache/. "
            f"A miss in offline/CI mode is a hard failure by design."
        )

    record = json.loads(path.read_text(encoding="utf-8"))
    return record["response"]
