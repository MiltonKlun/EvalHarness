"""Central configuration: env loading, pinned model snapshots, and the LIVE_LLM switch.

Everything that can drift in a non-deterministic eval harness is pinned in one place:
- the generator model (Gemini) and embedding model under test,
- the judge model (Claude) — a *different family*, so the evaluator never grades its
  own homework (see the build plan's independence property),
- the LIVE_LLM record/replay switch.

`require()` fails loud when a key is missing, so a misconfigured run errors immediately
instead of silently degrading.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present. Real CI sets env vars directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# --- Pinned model snapshots (see plan: "Pin exact model snapshot strings") ---------
# Generator = Gemini (system under test). Judge = Claude (independent evaluator).
# Claude Haiku 4.5 is the cheapest current Claude model, well-suited to LLM-as-judge
# and a deliberately different family from the Gemini generator.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "google_genai:gemini-3.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic:claude-haiku-4-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# --- Decode modes (exercised by the determinism experiment, plan 1.3 / 2.6) --------
# "max_pinned": pin every knob the API exposes (temp=0, top_p/top_k fixed, seed set)
#               and then *measure* the residual variance — we do not claim determinism.
# "near_det":   temperature=0 only.
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "0"))

# --- Record/replay switch ----------------------------------------------------------
# Truthy LIVE_LLM => real API calls (and record). Default => replay, hard-fail on miss.
LIVE_LLM = os.getenv("LIVE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}

CACHE_DIR = _REPO_ROOT / "evals" / "cache"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing — fail loud, never silently."""


def require(var: str) -> str:
    """Return env var ``var`` or raise ConfigError. Use at the point of need.

    Kept lazy (not validated at import) so that cached/offline CI runs, which need
    no API keys, don't fail just because secrets aren't present.
    """
    value = os.getenv(var)
    if not value:
        raise ConfigError(
            f"Required environment variable {var!r} is not set. "
            f"Copy .env.example to .env and fill it in, or export it in your shell."
        )
    return value
