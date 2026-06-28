"""
Embeddings via LangChain's HuggingFaceEmbeddings wrapper.

Uses sentence-transformers locally — same model as before (all-MiniLM-L6-v2),
but now exposed through the official langchain-huggingface integration so it
plugs directly into QdrantVectorStore without any manual batch handling.
"""
from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFaceEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
