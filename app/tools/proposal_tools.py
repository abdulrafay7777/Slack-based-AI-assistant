from langchain_core.tools import tool
from typing import Optional
from app.vector import retriever as vec

@tool
async def search_proposals(
    query: str,
    industry: Optional[str] = None,
    top_k: int = 4,
) -> list:
    """
    Search past proposals using semantic similarity.
    Use this multiple times with different queries to find context for each proposal section.
    
    Args:
        query:    What you're looking for (e.g. 'timeline phases for retail client')
        industry: Optional industry filter (e.g. 'retail', 'logistics')
        top_k:    Number of results (default 4, max 6)
    
    Returns:
        List of relevant chunks with text, section, source_file, score.
    """
    # Limit top_k to prevent token overflow
    top_k = min(top_k, 2)
    chunks = await vec.search(query=query, top_k=top_k, industry_filter=industry)
    return chunks