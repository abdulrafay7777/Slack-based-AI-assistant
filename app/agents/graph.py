"""
LangGraph graph definition.

Architecture:
  intake_node -> research_node -> writer_node
                                       ^
                   revision_node ------+  (targeted rewrites loop back to writer)

Routing:
  - After intake: if error -> END, else -> research
  - After research: if error -> END, else -> writer
  - After writer: END (Slack delivery happens in the API layer, not the graph)
  - Revision entry point skips intake+research and goes straight to writer
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from app.state import ProposalState
from app.agents.intake import intake_agent
from app.agents.research import research_agent
from app.agents.writer import writer_agent


# Conditional routing
def route_after_intake(state: ProposalState) -> Literal["research", "__end__"]:
    if state.get("error"):
        return END
    if not state.get("intake_complete"):
        return END
    return "research"


def route_after_research(state: ProposalState) -> Literal["writer", "__end__"]:
    if state.get("error"):
        return END
    if not state.get("research_complete"):
        return END
    return "writer"


# Graph builder
def build_proposal_graph() -> StateGraph:
    """
    Build and compile the full proposal generation graph.

    Entry points:
      - "intake"  : full pipeline (new transcript)
      - "writer"  : revision only (skip intake + research)
    """
    graph = StateGraph(ProposalState)

    graph.add_node("intake", intake_agent)
    graph.add_node("research", research_agent)
    graph.add_node("writer", writer_agent)
    graph.set_entry_point("intake")

    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"research": "research", END: END},
    )

    graph.add_conditional_edges(
        "research",
        route_after_research,
        {"writer": "writer", END: END},
    )

    graph.add_edge("writer", END)

    return graph.compile()


def build_revision_graph() -> StateGraph:
    """
    Lightweight graph for revision requests.
    Skips intake and research; goes directly to writer.
    """
    graph = StateGraph(ProposalState)

    graph.add_node("writer", writer_agent)
    graph.set_entry_point("writer")
    graph.add_edge("writer", END)

    return graph.compile()


# Compile once at import time
proposal_graph = build_proposal_graph()
revision_graph = build_revision_graph()