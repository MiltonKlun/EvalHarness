"""Build the FAISS vector store from the corpus — a deterministic fixture step.

Run this once (locally, with GOOGLE_API_KEY set) to (re)build ``app/vectorstore/``,
then commit the result. CI and other machines load the *committed* store and never
re-embed — so retrieval is identical everywhere and embedding drift can't silently
move "groundedness" numbers (plan 1.2).

Usage:
    python -m app.ingest          # build and persist
    python -m app.ingest --stats  # build, then print chunk stats and exit

Everything that affects the store is pinned in shared.config: the embedding model
snapshot, chunk size, and overlap. Changing any of them is a versioned change — rebuild
and re-commit.
"""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import config


def load_corpus() -> list[Document]:
    """Load every .md file in the corpus dir (except the provenance README)."""
    docs: list[Document] = []
    for path in sorted(config.CORPUS_DIR.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    if not docs:
        raise FileNotFoundError(f"No corpus documents found in {config.CORPUS_DIR}")
    return docs


def chunk(docs: list[Document]) -> list[Document]:
    """Split docs with the pinned strategy.

    The separator order keeps each markdown section together where possible (split on
    blank lines / headers before mid-sentence), so a single spec isn't cut in half.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_documents(docs)


def build():
    """Embed the chunked corpus and return a persisted FAISS store."""
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    config.require("GOOGLE_API_KEY")
    chunks = chunk(load_corpus())
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    store = FAISS.from_documents(chunks, embeddings)
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.VECTORSTORE_DIR))
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the corpus vector store.")
    parser.add_argument("--stats", action="store_true", help="print chunk stats and exit")
    args = parser.parse_args()

    if args.stats:
        chunks = chunk(load_corpus())
        sizes = [len(c.page_content) for c in chunks]
        print(f"documents: {len(load_corpus())}")
        print(f"chunks:    {len(chunks)}")
        print(
            f"chunk size: min={min(sizes)} max={max(sizes)} "
            f"(target {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP})"
        )
        by_source: dict[str, int] = {}
        for c in chunks:
            by_source[c.metadata["source"]] = by_source.get(c.metadata["source"], 0) + 1
        for src, n in sorted(by_source.items()):
            print(f"  {src}: {n} chunks")
        return

    build()
    print(f"Vector store written to {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
