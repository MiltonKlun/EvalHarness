"""The Claude judge for DeepEval — a different model family from the Gemini generator.

DeepEval's LLM-as-judge metrics (faithfulness, relevancy) need a model that can return
*structured* output against a Pydantic schema. We wrap Anthropic Claude (pinned to the
cheap Haiku tier) via ``instructor``, which enforces the schema for us.

Independence property (plan 2.2): the system under test is Gemini; the judge is Claude.
The judge never grades its own homework.

Cost: Haiku is the cheapest Claude model; judge prompts are tiny (one answer + its
context per call).

Record/replay: by default every judge call is routed through ``shared.cache`` exactly like
the generator and the toxicity judge. In replay mode a recorded verdict is returned with NO
Anthropic key and NO network — this is what lets a fresh keyless clone run the full suite
green. In live-record mode (``LIVE_LLM``) the real Claude call is made and recorded. DeepEval
issues several judge calls per ``measure()`` (claims, verdicts, reason); each caches
independently under its own prompt. ``config.require`` runs only inside the compute closure,
so replay never needs a key.

``config.JUDGE_LIVE`` overrides this: the cache is bypassed entirely (no read, no write) and
a fresh judgement is returned. That is the judged CI tier's second-opinion mode — see
``shared.config``.
"""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from shared import cache, config

# Strip the "anthropic:" provider prefix from the pinned JUDGE_MODEL for the raw SDK.
_JUDGE_MODEL_ID = config.JUDGE_MODEL.split(":", 1)[-1]
_MAX_TOKENS = 1024


class ClaudeJudge(DeepEvalBaseLLM):
    """A DeepEval judge backed by Claude Haiku, routed through the record/replay cache."""

    def __init__(self, model_id: str = _JUDGE_MODEL_ID) -> None:
        self.model_id = model_id

    def load_model(self):
        # DeepEval calls this; we have no persistent client (the cache/compute owns it).
        return None

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        """Return the judge's output for ``prompt``.

        Two shapes, distinguished in the cache key so the same prompt can't collide:
          - schema=None  -> raw text; cached and replayed as the plain string.
          - schema given -> a Pydantic instance; cached as JSON, rebuilt on replay.

        When ``config.JUDGE_LIVE`` is set the cache is bypassed entirely (no read, no write):
        a fresh live judgement is returned. That is the judged CI tier's mode — a genuine
        second opinion that never overwrites the committed verdict baseline (recording stays
        LIVE_LLM's job). Otherwise the call is routed through the record/replay cache.
        """
        params = {"schema": schema.__name__ if schema else "raw", "max_tokens": _MAX_TOKENS}

        def _compute() -> str:
            # Import + key check happen ONLY on a real (live) call, never on replay.
            import instructor
            from anthropic import Anthropic

            config.require("ANTHROPIC_API_KEY")
            client = Anthropic()
            if schema is None:
                resp = client.messages.create(
                    model=self.model_id,
                    max_tokens=_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            obj = instructor.from_anthropic(client).messages.create(
                model=self.model_id,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
            )
            return obj.model_dump_json()

        # JUDGE_LIVE: fresh call, no cache read/write. Otherwise: record/replay.
        raw = (
            _compute()
            if config.JUDGE_LIVE
            else cache.cached_call(config.JUDGE_MODEL, prompt, params, _compute)
        )
        if schema is None:
            return raw
        return schema.model_validate_json(raw)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        # DeepEval accepts a sync fallback; our suite runs synchronously.
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Claude judge ({self.model_id})"
