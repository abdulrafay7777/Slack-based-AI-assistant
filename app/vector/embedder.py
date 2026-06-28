"""
Thin wrapper around local HuggingFace embeddings using SentenceTransformers.
"""
from __future__ import annotations

import os
import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _embed_text_sync(text: str) -> List[float]:
    model = _get_model()
    text = text.replace("\n", " ").strip()
    return model.encode(text).tolist()

def _embed_batch_sync(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    cleaned = [t.replace("\n", " ").strip() for t in texts]
    embeddings = model.encode(cleaned)
    return embeddings.tolist()

async def embed_text(text: str) -> List[float]:
    """Embed a single string. Returns a list of floats."""
    return await asyncio.to_thread(_embed_text_sync, text)

async def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple strings in one call."""
    return await asyncio.to_thread(_embed_batch_sync, texts)
