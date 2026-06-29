"""Retrieval interface: ``retrieve(query) -> list[Chunk]``.

Loads the *committed* FAISS store (built by ``app.ingest``) and returns the top-k most
relevant chunks, deduplicated, each carrying its source metadata. This clean boundary
is what the eval suite asserts against (right chunks, right sources) — see plan 1.2.

Retrieval is routed through the SAME record/replay cache as LLM calls:
  - LIVE_LLM=1 -> embed the query (needs GOOGLE_API_KEY) and record the chunks,
  - default    -> replay recorded chunks from disk with NO key (this is what lets the
                  fast CI tier run the deterministic eval checks completely keyless).

A cache miss in replay mode is a hard error, exactly like the LLM cache — a new query
that was never recorded must fail loudly, not silently reach for the network.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.schema import Chunk
from shared import cache, config


@lru_cache(maxsize=1)
def _load_store():
    """Load and cache the committed FAISS store. Fails loud if it hasn't been built."""
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not (config.VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"No vector store at {config.VECTORSTORE_DIR}. "
            f"Build it once with `python -m app.ingest` (needs GOOGLE_API_KEY), "
            f"then commit app/vectorstore/."
        )
    config.require("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        str(config.VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # our own committed store, not untrusted
    )


def _retrieve_live(query: str, k: int) -> list[Chunk]:
    """Embed the query against the committed store (needs GOOGLE_API_KEY)."""
    store = _load_store()
    # FAISS returns (Document, distance); lower distance = more similar.
    hits = store.similarity_search_with_score(query, k=k)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for doc, distance in hits:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)
        chunks.append(
            Chunk(
                text=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                # Convert distance to a "higher = better" score for readability.
                score=float(-distance),
            )
        )
    return chunks


def retrieve(query: str, k: int | None = None) -> list[Chunk]:
    """Return the top-k most relevant corpus chunks for ``query``.

    Routed through the record/replay cache: replay returns recorded chunks with no key;
    live mode embeds + records. Chunks carry source + score for downstream checks.
    """
    k = k or config.RETRIEVAL_TOP_K
    # The cache stores strings, so we (de)serialize the chunk list as JSON. The cache key
    # uses a "retriever:<embedding-model>" pseudo-provider + params so it's distinct from
    # LLM recordings and invalidates if the embedding model or k changes.
    provider = f"retriever:{config.EMBEDDING_MODEL}"

    def _compute() -> str:
        chunks = _retrieve_live(query, k)
        return json.dumps([{"text": c.text, "source": c.source, "score": c.score} for c in chunks])

    raw = cache.cached_call(provider, query, {"k": k}, _compute)
    return [
        Chunk(text=d["text"], source=d["source"], score=d.get("score")) for d in json.loads(raw)
    ]
