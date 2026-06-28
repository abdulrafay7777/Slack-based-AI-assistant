"""
Ingestion pipeline — converts chunk dicts into LangChain Documents and
upserts them into the QdrantVectorStore.

Why LangChain Documents:
  - Standard interface understood by QdrantVectorStore.add_documents()
  - Metadata dict is preserved as-is in the Qdrant payload under "metadata"
  - No manual PointStruct or UUID handling required
"""
from __future__ import annotations

import asyncio
from typing import List, Dict, Any

from langchain_core.documents import Document

from app.vector.store import get_vector_store


async def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Insert or update chunks into the vector store.

    Each chunk dict must have:
        text       : str - the text content
        source_file: str - filename of the source proposal
        industry   : str - normalised industry tag (e.g. "retail")
        section    : str - proposal section name (e.g. "executive_summary")
        chunk_index: int - position within the source document
    """
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "source_file": c["source_file"],
                "industry": c["industry"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
            },
        )
        for c in chunks
    ]
    store = get_vector_store()
    # QdrantVectorStore.add_documents is sync; run in thread to stay async-safe
    await asyncio.to_thread(store.add_documents, docs)
