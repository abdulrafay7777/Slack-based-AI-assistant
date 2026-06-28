"""
Agent 2 - Research

Uses LangGraph's create_react_agent so the LLM decides:
  - what to search for
  - how many times to search
  - whether to filter by industry or search broadly

This is proper tool-calling RAG, not hardcoded queries.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.state import ProposalState
from app.tools.proposal_tools import search_proposals

_SYSTEM_PROMPT = """You are the Research Agent for a business proposal system.

You have a search tool that queries a knowledge base of past proposals using semantic search.

Given the client data, your job is to run MULTIPLE targeted searches to gather relevant 
context for each section of the proposal. Run separate searches for:

1. The client's core problem and industry context
2. Proposed solution approaches for their type of problem
3. Timeline and project phases
4. Budget ranges and pricing models
5. Next steps and onboarding approaches
6. Executive summary framing

Rules:
- Always try an industry-filtered search first, then a broad search if results are thin
- Run exactly 2 or 3 searches covering the most critical sections
- Do not run more than 3 searches to avoid exceeding context token limits
- You are done when you have gathered context from these searches
"""

def _build_agent():
    llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return create_react_agent(
        model=llm,
        tools=[search_proposals],
        prompt=_SYSTEM_PROMPT,
    )

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


async def research_agent(state: ProposalState) -> dict[str, Any]:
    client_data = state.get("client_data")
    if not client_data:
        return {
            "error": "Research agent requires completed intake data.",
            "status": "error",
        }

    result = await _get_agent().ainvoke({
        "messages": [
            HumanMessage(content=f"Client data:\n{json.dumps(client_data, indent=2)}\n\nSearch for relevant proposal context now.")
        ]
    })

    # Extract all tool result messages and parse the chunks out
    chunks = []
    seen: set[str] = set()

    for msg in result["messages"]:
        # Tool results come back as ToolMessage with JSON list content
        if hasattr(msg, "content") and msg.__class__.__name__ == "ToolMessage":
            try:
                parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(parsed, list):
                    for chunk in parsed:
                        key = chunk.get("text", "")[:120]
                        if key not in seen:
                            seen.add(key)
                            chunks.append(chunk)
            except (json.JSONDecodeError, TypeError):
                pass

    chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_chunks = chunks[:30]

    return {
        "retrieved_chunks": top_chunks,
        "research_complete": True,
        "status": "research_done",
        "error": None,
    }