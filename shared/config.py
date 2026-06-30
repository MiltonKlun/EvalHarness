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

# Graceful-degradation guard, applied at IMPORT time (before LangChain reads these).
# LangChain arms tracing from LANGSMITH_TRACING very early; if tracing is requested but
# no key is present, neutralise it here so a missing/blank key can never cause noisy
# 401 export errors mid-run. LangSmith is a bonus layer, never load-bearing.
if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
} and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"

# --- Pinned model snapshots (see plan: "Pin exact model snapshot strings") ---------
# Generator = Gemini (system under test). Judge = Claude (independent evaluator).
# Claude Haiku 4.5 is the cheapest current Claude model, well-suited to LLM-as-judge
# and a deliberately different family from the Gemini generator.
# gemini-2.5-flash chosen over the newer 3.x flash for the more workable free-tier
# generate_content limit. Measured live, 2.5-flash gives ~20 generate-req/day on the AI
# Studio free tier (the API 429 reports `limit: 20`); that hard cap is why record/replay
# exists — see docs/COST.md. Revisit if billing is on.
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "google_genai:gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic:claude-haiku-4-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# --- Retrieval / chunking (a drift surface — pin it, plan 1.2) ---------------------
# Chunk size/overlap are documented, deliberate choices: small enough to be precise,
# large enough that a single spec (e.g. one drone's stats) is not split mid-fact.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
# top-k is tuned, not left at a library default: the corpus is small and a question
# rarely needs more than the few most-relevant passages.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

CORPUS_DIR = _REPO_ROOT / "app" / "corpus"
VECTORSTORE_DIR = _REPO_ROOT / "app" / "vectorstore"

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


def langsmith_enabled() -> bool:
    """Whether LangSmith tracing is active (tracing on AND a key present).

    The actual neutralisation of a key-less ``LANGSMITH_TRACING=true`` happens at import
    time above (before LangChain reads the flag); this is just the query.
    """
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } and bool(os.getenv("LANGSMITH_API_KEY"))


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
