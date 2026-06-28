"""
Vector store singleton using langchain-qdrant's QdrantVectorStore.

Responsibilities:
  - Create and cache the QdrantClient (sync)
  - Ensure the Qdrant collection exists before first use
  - Expose get_vector_store() for ingestion and retrieval modules
  - Expose collection_count() for startup health-check in main.py

Why sync QdrantClient:
  langchain-qdrant's QdrantVectorStore wraps a sync QdrantClient by default.
  All async callers (ingestion.py, retrieval.py) use asyncio.to_thread() to
  keep the event loop unblocked.
"""
from __future__ import annotations

import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.vector.embedder import get_embeddings, EMBEDDING_DIM

COLLECTION_NAME = "proposals"
QDRANT_URL = os.getenv("QDRANT_URL", "data/qdrant_db")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


def _get_client() -> QdrantClient:
    """Return a singleton QdrantClient (sync) based on QDRANT_URL."""
    global _client
    if _client is None:
        if QDRANT_URL == ":memory:":
            _client = QdrantClient(location=":memory:")
        elif QDRANT_URL.startswith("http://") or QDRANT_URL.startswith("https://"):
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            os.makedirs(QDRANT_URL, exist_ok=True)
            _client = QdrantClient(path=QDRANT_URL)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection and payload index if they don't exist."""
    from qdrant_client.models import PayloadSchemaType

    names = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        # Index metadata.industry for fast keyword filtering
        # langchain-qdrant stores Document metadata under the "metadata" key
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.industry",
            field_schema=PayloadSchemaType.KEYWORD,
        )


def get_vector_store() -> QdrantVectorStore:
    """
    Return a singleton QdrantVectorStore, creating the collection if needed.

    Used by both ingestion.py (add_documents) and retrieval.py
    (similarity_search_with_score).
    """
    global _vector_store
    if _vector_store is None:
        client = _get_client()
        _ensure_collection(client)
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
        )
    return _vector_store


async def collection_count() -> int:
    """Return the number of stored chunks (used by main.py at startup)."""
    client = _get_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count or 0
    except Exception:
        return 0
