"""
Agent 1 - Intake

Single responsibility:
  Read the raw transcript, extract structured client information, and flag
  any fields that are absent or ambiguous. Never guess missing values.

Output written to ProposalState:
  client_data, missing_fields, intake_complete
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.state import ProposalState

_SYSTEM_PROMPT = """You are the Intake Agent for a business proposal system.

Your ONLY job is to read a client discovery call transcript and extract the following fields:
  - company_name    : Name of the client company
  - industry        : Industry/sector (e.g. retail, logistics, healthcare)
  - problem         : The core business problem or pain point they described
  - goals           : List of explicit goals or success criteria they mentioned
  - budget          : Any budget figure, range, or constraint mentioned
  - timeline        : Any deadline, delivery window, or urgency mentioned
  - stakeholders    : Names/roles of people mentioned as decision-makers or influencers

Rules:
1. ONLY extract what is explicitly stated in the transcript.
2. If a field is missing or ambiguous, set it to null do NOT guess or infer.
3. Return ONLY valid JSON, no explanation, no markdown fences.
4. The JSON must have exactly two keys:
   "client_data": {{ ... fields above ... }}
   "missing_fields": [ list of field names that are null or unclear ]

Example output:
{{
  "client_data": {{
    "company_name": "Acme Corp",
    "industry": "retail",
    "problem": "High cart abandonment rate on mobile",
    "goals": ["Reduce abandonment by 20%", "Improve mobile UX"],
    "budget": null,
    "timeline": "Q3 2025",
    "stakeholders": ["Jane Smith (CTO)", "Mark Lee (Head of Product)"]
  }},
  "missing_fields": ["budget"]
}}
"""


async def intake_agent(state: ProposalState) -> dict[str, Any]:
    """
    LangGraph node function for the Intake Agent.
    Reads state["transcript_raw"], returns updated state keys.
    """
    transcript = state.get("transcript_raw", "")

    if not transcript:
        return {
            "error": "No transcript available. Please upload a transcript first.",
            "status": "error",
        }

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"TRANSCRIPT:\n\n{transcript}"),
    ]

    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        client_data = parsed.get("client_data", {})
        missing_fields = parsed.get("missing_fields", [])
    except json.JSONDecodeError as e:
        return {
            "error": f"Intake agent returned invalid JSON: {e}\nRaw: {raw[:200]}",
            "status": "error",
        }

    return {
        "client_data": client_data,
        "missing_fields": missing_fields,
        "intake_complete": True,
        "status": "intake_done",
        "error": None,
    }
