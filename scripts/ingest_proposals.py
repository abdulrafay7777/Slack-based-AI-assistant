"""
Startup ingestion script.

Reads every .txt file in data/proposals/, chunks it using LangChain's
RecursiveCharacterTextSplitter, tags each chunk with metadata
(industry, section, source_file), and upserts into the Qdrant vector store.

Chunking strategy:
  - RecursiveCharacterTextSplitter (official LangChain splitter)
  - chunk_size=2000 chars, chunk_overlap=200 chars
  - Splits on paragraphs → sentences → words in order, preserving context

Run once at startup (app/main.py calls ingest_all() automatically if
the collection is empty).
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vector.ingestion import upsert_chunks

DATA_DIR = Path(os.getenv("PROPOSALS_DIR", "data/proposals"))

CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200

# Industry tags derived from filename keywords
_INDUSTRY_MAP = {
    "logistics":      "logistics",
    "retail":         "retail",
    "healthcare":     "healthcare",
    "manufacturing":  "manufacturing",
    "fintech":        "fintech",
    "construction":   "construction",
    "distribution":   "distribution",
    "consulting":     "consulting",
    "fleet":          "fleet",
}

# Section heading patterns to detect proposal structure
_SECTION_PATTERNS = [
    (re.compile(r"(executive\s+summary)", re.I),        "executive_summary"),
    (re.compile(r"(proposed\s+solution|our\s+approach)", re.I), "proposed_solution"),
    (re.compile(r"(timeline|project\s+phases|milestones)", re.I), "timeline"),
    (re.compile(r"(budget|pricing|investment|cost)", re.I), "budget"),
    (re.compile(r"(next\s+steps|action\s+items)", re.I), "next_steps"),
    (re.compile(r"(cover|introduction|overview)", re.I), "cover"),
]

# Shared splitter instance
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    length_function=len,
)


def _detect_industry(filename: str) -> str:
    lower = filename.lower()
    for keyword, tag in _INDUSTRY_MAP.items():
        if keyword in lower:
            return tag
    return "general"


def _detect_section(text: str) -> str:
    for pattern, label in _SECTION_PATTERNS:
        if pattern.search(text[:200]):   # check the start of the chunk
            return label
    return "general"


def _chunk_document(text: str, source_file: str, industry: str) -> List[Dict[str, Any]]:
    """
    Chunk a proposal document using RecursiveCharacterTextSplitter and
    return a list of chunk dicts ready for upsert.
    """
    raw_chunks = _splitter.split_text(text)
    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "source_file": source_file,
            "industry": industry,
            "section": _detect_section(chunk_text),
            "chunk_index": idx,
        })
    return chunks


async def ingest_file(filepath: Path) -> int:
    """Ingest a single proposal file. Returns number of chunks upserted."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            print(f"  [SKIP] {filepath.name} is empty.")
            return 0

        # Check file size before processing
        if len(text) > 5_000_000:  # 5MB limit
            print(f"  [ERROR] {filepath.name} is too large ({len(text):,} chars). Skipping.")
            return 0

        industry = _detect_industry(filepath.name)
        chunks = _chunk_document(text, source_file=filepath.name, industry=industry)

        if not chunks:
            print(f"  [SKIP] {filepath.name} produced no chunks.")
            return 0

        await upsert_chunks(chunks)
        print(f"  [OK]   {filepath.name:40s}  industry={industry:15s}  chunks={len(chunks)}")
        return len(chunks)

    except MemoryError:
        print(f"  [ERROR] {filepath.name} caused MemoryError. File too large. Skipping.")
        return 0
    except Exception as e:
        print(f"  [ERROR] {filepath.name} failed: {e}")
        return 0


async def ingest_all() -> None:
    """Ingest all .txt files in DATA_DIR into the vector store."""
    files = sorted(DATA_DIR.glob("*.txt"))
    # Exclude the test client transcript from being ingested as a past proposal
    files = [f for f in files if "client_transcript.txt" not in f.name]

    if not files:
        print(f"No .txt files found in {DATA_DIR}. Check PROPOSALS_DIR env var.")
        return

    print(f"Ingesting {len(files)} proposal files from {DATA_DIR}…")
    total = 0
    for f in files:
        total += await ingest_file(f)

    print(f"\n Ingestion complete. Total chunks: {total}")


if __name__ == "__main__":
    asyncio.run(ingest_all())
