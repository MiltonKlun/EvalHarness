"""Retrieval interface: ``retrieve(query) -> list[Chunk]``.

Loads the *committed* FAISS store (built by ``app.ingest``) and returns the top-k most
relevant chunks, deduplicated, each carrying its source metadata. This clean boundary
is what the eval suite asserts against (right chunks, right sources) — see plan 1.2.

Loading the store still needs the embedding model to embed the *query*, so a live
GOOGLE_API_KEY is required at retrieve time. (The corpus embeddings themselves are
already baked into the committed store and are never recomputed.)
"""

from __future__ import annotations

from functools import lru_cache

from app.schema import Chunk
from shared import config


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


def retrieve(query: str, k: int | None = None) -> list[Chunk]:
    """Return the top-k most relevant corpus chunks for ``query``.

    Chunks are deduplicated by text (FAISS can surface near-identical overlapping
    passages) and carry their source filename + similarity score so downstream
    groundedness/citation checks have what they need.
    """
    k = k or config.RETRIEVAL_TOP_K
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
