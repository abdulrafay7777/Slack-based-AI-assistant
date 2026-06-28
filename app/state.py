from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages


class ClientData(TypedDict, total=False):
    company_name: Optional[str]
    industry: Optional[str]
    problem: Optional[str]
    goals: Optional[List[str]]
    budget: Optional[str]
    timeline: Optional[str]
    stakeholders: Optional[List[str]]


class ProposalState(TypedDict):
    # Session identity
    session_id: str
    slack_user_id: str
    slack_channel_id: str

    # Raw inputs
    transcript_raw: Optional[str]

    # Intake agent output
    client_data: Optional[ClientData]
    missing_fields: Optional[List[str]]
    intake_complete: bool

    # Research agent output
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    research_complete: bool

    # Writer agent output
    proposal_sections: Optional[List[Dict[str, str]]]
    docx_path: Optional[str]
    writer_complete: bool

    # Revision tracking
    revision_request: Optional[str]
    target_section: Optional[str]
    revision_count: int

    # Conversation history (LangGraph managed)
    messages: Annotated[List, add_messages]

    # Error / status
    error: Optional[str]
    status: str                            # "pending" | "processing" | "awaiting_review" | "done" | "error"
