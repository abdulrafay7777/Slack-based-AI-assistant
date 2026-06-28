"""
Startup ingestion script.

Reads every .txt file in data/proposals/, chunks it intelligently,
tags each chunk with metadata (industry, section, source_file), and
upserts into the Qdrant vector store.

Chunking strategy:
  - Split by detected section headings first (structural chunking)
  - Then by fixed token window (512 tokens) with 64-token overlap for
    sections that are too long
  - This preserves semantic context better than pure token chunking

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

# Allow running directly: python -m scripts.ingest_proposals
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vector.retriever import upsert_chunks, ensure_collection

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


def _split_by_headings(text: str) -> List[tuple[str, str]]:
    """
    Split document text by markdown-style or ALL-CAPS headings.
    Returns list of (heading, body) tuples.
    """
    # Match lines that look like headings: ALL CAPS, Title Case standalone line, or ## Markdown
    heading_re = re.compile(
        r"^(?:#{1,3}\s+.+|[A-Z][A-Z\s]{3,}|[A-Z][a-z][\w\s]{2,}:)\s*$",
        re.MULTILINE,
    )
    positions = [(m.start(), m.end(), m.group()) for m in heading_re.finditer(text)]

    if not positions:
        return [("", text)]

    sections = []
    for i, (start, end, heading) in enumerate(positions):
        body_start = end
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((heading.strip(), body))

    # Include content before first heading
    if positions[0][0] > 0:
        preamble = text[:positions[0][0]].strip()
        if preamble:
            sections.insert(0, ("", preamble))

    return sections


def _fixed_window_chunks(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Slide a fixed window over text when a section is too long."""
    # Prevent memory issues with extremely large text
    if len(text) > 1_000_000:  # 1MB text limit
        print(f"    [WARNING] Text too large ({len(text)} chars), truncating to 1M chars")
        text = text[:1_000_000]
    
    chunks = []
    start = 0
    max_iterations = 10_000  # Safety limit to prevent infinite loops
    iteration = 0
    
    while start < len(text) and iteration < max_iterations:
        iteration += 1
        end = min(start + size, len(text))
        
        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + size // 2:
                end = boundary + 1
        
        chunk_text = text[start:end].strip()
        if chunk_text:  # Only add non-empty chunks
            chunks.append(chunk_text)
        
        # Move start forward, ensuring progress
        new_start = end - overlap
        if new_start <= start:  # Prevent infinite loop
            new_start = start + 1
        start = new_start
        
        if start >= len(text):
            break
    
    if iteration >= max_iterations:
        print(f"    [WARNING] Hit max iterations ({max_iterations}) while chunking")
    
    return [c for c in chunks if len(c) > 50]


def _chunk_document(text: str, source_file: str, industry: str) -> List[Dict[str, Any]]:
    """
    Chunk a proposal document and return a list of chunk dicts ready for upsert.
    """
    sections = _split_by_headings(text)
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for heading, body in sections:
        combined = f"{heading}\n{body}".strip() if heading else body
        section_tag = _detect_section(combined)

        if len(combined) <= CHUNK_SIZE_CHARS:
            chunks.append({
                "text": combined,
                "source_file": source_file,
                "industry": industry,
                "section": section_tag,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        else:
            # Section too long – apply fixed window
            sub_chunks = _fixed_window_chunks(combined)
            for sub in sub_chunks:
                chunks.append({
                    "text": sub,
                    "source_file": source_file,
                    "industry": industry,
                    "section": section_tag,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

    return chunks


async def ingest_file(filepath: Path) -> int:
    """Ingest a single proposal file. Returns number of chunks upserted."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            print(f"  [SKIP] {filepath.name} is empty.")
            return 0
        
        # Check file size before processing
        file_size = len(text)
        if file_size > 5_000_000:  # 5MB limit
            print(f"  [ERROR] {filepath.name} is too large ({file_size:,} chars). Skipping.")
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
    await ensure_collection()

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
