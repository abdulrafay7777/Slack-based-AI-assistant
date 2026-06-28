"""
Vector search using Qdrant (local in-memory for dev, remote for prod).

Why Qdrant:
- Native metadata filtering alongside vector search (no post-filter needed)
- Fast local mode (in-memory) for development, same API for prod
- Strong Python async client
- No external DB dependency for initial setup (unlike pgvector which needs Postgres extension)

Collection schema:
  payload: { source_file, industry, section, chunk_index, text }
"""
from __future__ import annotations

import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

from app.vector.embedder import embed_text, embed_batch, EMBEDDING_DIM

COLLECTION_NAME = "proposals"
QDRANT_URL = os.getenv("QDRANT_URL", "data/qdrant_db")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

_client: AsyncQdrantClient | None = None


def _get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL == ":memory:":
            _client = AsyncQdrantClient(location=":memory:")
        elif QDRANT_URL.startswith("http://") or QDRANT_URL.startswith("https://"):
            _client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            os.makedirs(QDRANT_URL, exist_ok=True)
            _client = AsyncQdrantClient(path=QDRANT_URL)
    return _client


async def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    from qdrant_client.models import PayloadSchemaType
    
    client = _get_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        # Create keyword index for industry field to enable filtering
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="industry",
            field_schema=PayloadSchemaType.KEYWORD,
        )


async def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Insert or update chunks into the vector store.

    Each chunk dict must have:
        text       : str   – the text content
        source_file: str   – filename of the source proposal
        industry   : str   – normalised industry tag (e.g. "retail")
        section    : str   – proposal section name (e.g. "executive_summary")
        chunk_index: int   – position within the source document
    """
    await ensure_collection()
    client = _get_client()

    texts = [c["text"] for c in chunks]
    vectors = await embed_batch(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "industry": chunk["industry"],
                "section": chunk["section"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    await client.upsert(collection_name=COLLECTION_NAME, points=points)


async def search(
    query: str,
    top_k: int = 8,
    industry_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search over the proposals collection.

    Args:
        query:           Natural language query string.
        top_k:           Number of results to return.
        industry_filter: Optional industry tag to restrict results (metadata filter).

    Returns:
        List of dicts with keys: text, source_file, industry, section, score
    """
    await ensure_collection()
    client = _get_client()

    query_vector = await embed_text(query)

    qdrant_filter = None
    if industry_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="industry",
                    match=MatchValue(value=industry_filter.lower()),
                )
            ]
        )

    # Use query_points for qdrant-client >= 1.18
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload["text"],
            "source_file": hit.payload.get("source_file", ""),
            "industry": hit.payload.get("industry", ""),
            "section": hit.payload.get("section", ""),
            "chunk_index": hit.payload.get("chunk_index", 0),
            "score": hit.score,
        }
        for hit in results.points
    ]


async def collection_count() -> int:
    """Return how many chunks are currently stored. Returns 0 if collection doesn't exist yet."""
    client = _get_client()
    try:
        info = await client.get_collection(COLLECTION_NAME)
        return info.points_count or 0
    except (ValueError, Exception):
        return 0
