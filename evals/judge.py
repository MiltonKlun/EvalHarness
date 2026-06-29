"""The Claude judge for DeepEval — a different model family from the Gemini generator.

DeepEval's LLM-as-judge metrics (faithfulness, relevancy) need a model that can return
*structured* output against a Pydantic schema. We wrap Anthropic Claude (pinned to the
cheap Haiku tier) via ``instructor``, which enforces the schema for us.

Independence property (plan 2.2): the system under test is Gemini; the judge is Claude.
The judge never grades its own homework.

Cost: Haiku is the cheapest Claude model; judge prompts are tiny (one answer + its
context per call). Combined with the record/replay cache, offline CI spends $0 on the
judge — only live runs cost anything.
"""

from __future__ import annotations

from anthropic import Anthropic
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from shared import config

# Strip the "anthropic:" provider prefix from the pinned JUDGE_MODEL for the raw SDK.
_JUDGE_MODEL_ID = config.JUDGE_MODEL.split(":", 1)[-1]


class ClaudeJudge(DeepEvalBaseLLM):
    """A DeepEval judge backed by Claude Haiku via instructor (schema-enforced output)."""

    def __init__(self, model_id: str = _JUDGE_MODEL_ID) -> None:
        self.model_id = model_id
        self._client = None  # lazy: don't construct a client unless a judge call happens

    def load_model(self):
        if self._client is None:
            config.require("ANTHROPIC_API_KEY")
            self._client = Anthropic()
        return self._client

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        import instructor

        client = instructor.from_anthropic(self.load_model())
        if schema is None:
            # Some metrics call without a schema; return raw text.
            resp = self.load_model().messages.create(
                model=self.model_id,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        return client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
        )

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        # DeepEval accepts a sync fallback; our suite runs synchronously.
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Claude judge ({self.model_id})"
