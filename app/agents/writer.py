"""
Agent 3 Writer

Single responsibility:
  Synthesise the Intake Agent's structured client data and the Research
  Agent's retrieved proposal chunks into a complete, section-by-section
  proposal. Also handles targeted revisions without rewriting the whole doc.

Output written to ProposalState:
  proposal_sections, docx_path, writer_complete
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import ProposalState
from app.docx_gen.builder import build_docx

_SYSTEM_PROMPT = """You are the Writer Agent for a professional consulting proposal system.

You will be given:
1. Structured client information (company, industry, problem, goals, budget, timeline, stakeholders)
2. Relevant excerpts from past proposals as reference context

Your job is to write a complete, tailored business proposal. You must decide on the most logical sections based on the retrieved context and client data.
You MUST include the following required sections: cover, executive summary, proposed solution, timeline, budget range, and next steps.
Always include a "cover" section as the first section.

Rules:
1. Write in a professional but human tone. Do not use corporate jargon.
2. Tailor every section to the specific client - do not use generic filler.
3. Use the reference context to inform structure, tone, and approach - but do not copy it verbatim.
4. Return ONLY valid JSON representing an array of section objects. No markdown, no preamble.
5. Each object must have "id" (a short string like "cover", "exec_summary"), "title" (the display title), and "content" (the text content).

Output format:
[
  {
    "id": "cover",
    "title": "Cover",
    "content": "..."
  },
  {
    "id": "executive_summary",
    "title": "Executive Summary",
    "content": "..."
  }
]
"""

_REVISION_SYSTEM_PROMPT = """You are the Writer Agent for a professional consulting proposal system.

You will be given the CURRENT full proposal (as a JSON array of section objects) and a REVISION REQUEST.

Your job:
1. Identify which section(s) the revision targets.
2. Rewrite ONLY those sections according to the request.
3. Return ONLY valid JSON containing the FULL proposal array including the updated section(s).
   Do not change sections that were not targeted.

Return ONLY valid JSON, no markdown, no explanation.
"""


def _format_context(chunks: list) -> str:
    """Format retrieved chunks as a readable context block."""
    if not chunks:
        return "No past proposal context available."
    lines = []
    for i, chunk in enumerate(chunks[:20], 1):   # cap at 20 chunks for context window
        lines.append(
            f"[Ref {i} | {chunk.get('source_file','')} | "
            f"{chunk.get('section','')} | score={chunk.get('score',0):.2f}]\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(lines)


async def writer_agent(state: ProposalState) -> dict[str, Any]:
    """
    LangGraph node for the Writer Agent.

    Handles two modes:
      - Full generation: when proposal_sections is empty / None
      - Targeted revision: when revision_request is set
    """
    client_data = state.get("client_data") or {}
    retrieved_chunks = state.get("retrieved_chunks") or []
    session_id = state.get("session_id", "unknown")
    revision_request = state.get("revision_request")
    existing_sections = state.get("proposal_sections")

    llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.1-70b-versatile"),
        temperature=float(os.getenv("WRITER_TEMPERATURE", "0.4")),
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # MODE 1: Targeted revision
    if revision_request and existing_sections:
        messages = [
            SystemMessage(content=_REVISION_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"CURRENT PROPOSAL:\n{json.dumps(existing_sections, indent=2)}\n\n"
                f"REVISION REQUEST:\n{revision_request}"
            )),
        ]
        response = await llm.ainvoke(messages)
        raw = _strip_fences(response.content.strip())

        try:
            updated_sections = json.loads(raw)
        except json.JSONDecodeError as e:
            return {
                "error": f"Writer revision returned invalid JSON: {e}",
                "status": "error",
            }

        docx_path = await build_docx(
            sections=updated_sections,
            client_data=client_data,
            session_id=session_id,
            revision=state.get("revision_count", 0),
        )

        return {
            "proposal_sections": updated_sections,
            "docx_path": docx_path,
            "writer_complete": True,
            "revision_request": None,
            "target_section": None,
            "status": "done",
            "error": None,
        }

    # MODE 2: Full generation
    context_text = _format_context(retrieved_chunks)

    user_content = (
        f"CLIENT DATA:\n{json.dumps(client_data, indent=2)}\n\n"
        f"REFERENCE CONTEXT FROM PAST PROPOSALS:\n{context_text}"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = await llm.ainvoke(messages)
    raw = _strip_fences(response.content.strip())

    try:
        sections = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "error": f"Writer agent returned invalid JSON: {e}\nRaw: {raw[:300]}",
            "status": "error",
        }

    # Validate it's a list
    if not isinstance(sections, list):
        return {
            "error": "Writer agent did not return a JSON array",
            "status": "error",
        }

    docx_path = await build_docx(
        sections=sections,
        client_data=client_data,
        session_id=session_id,
        revision=0,
    )

    return {
        "proposal_sections": sections,
        "docx_path": docx_path,
        "writer_complete": True,
        "status": "done",
        "error": None,
    }


def _strip_fences(text: str) -> str:
    """Remove ```json / ``` fences if the model adds them."""
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content between fences
        content = parts[1] if len(parts) > 1 else text
        if content.startswith("json"):
            content = content[4:]
        return content.strip()
    return text
