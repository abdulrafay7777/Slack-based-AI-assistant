"""
Retrieval pipeline — semantic search over the proposals vector store.

Uses QdrantVectorStore.similarity_search_with_score() from langchain-qdrant,
which handles query embedding, Qdrant search, and result unpacking internally.

Metadata filter note:
  langchain-qdrant stores Document metadata under the "metadata" key in the
  Qdrant payload, so filters must use "metadata.<field>" as the key path.
"""
from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.vector.store import get_vector_store


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
        industry_filter: Optional industry tag to restrict results.

    Returns:
        List of dicts with keys: text, source_file, industry, section,
        chunk_index, score.
    """
    store = get_vector_store()

    qdrant_filter = None
    if industry_filter:
        # langchain-qdrant stores metadata under the "metadata" payload key
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.industry",
                    match=MatchValue(value=industry_filter.lower()),
                )
            ]
        )

    # similarity_search_with_score is sync; run in thread to stay async-safe
    results = await asyncio.to_thread(
        store.similarity_search_with_score,
        query,
        k=top_k,
        filter=qdrant_filter,
    )

    return [
        {
            "text": doc.page_content,
            "source_file": doc.metadata.get("source_file", ""),
            "industry": doc.metadata.get("industry", ""),
            "section": doc.metadata.get("section", ""),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score": score,
        }
        for doc, score in results
    ]
