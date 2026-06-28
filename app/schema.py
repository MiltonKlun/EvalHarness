"""Shared data shapes for the RAG app.

A ``Chunk`` carries the retrieved text *plus the source metadata* that downstream
groundedness and citation checks need (plan 1.2 RAG-correctness checklist). Keeping
this explicit — rather than passing bare strings around — is what lets the eval suite
assert "the answer cites the right source", not just "the answer looks right".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrieved passage with provenance.

    Attributes:
        text: the chunk's text content.
        source: the corpus filename this chunk came from (e.g. "02_products.md").
        score: retrieval similarity score (higher = more relevant). Optional;
            populated by the retriever, ``None`` when not applicable.
    """

    text: str
    source: str
    score: float | None = None
