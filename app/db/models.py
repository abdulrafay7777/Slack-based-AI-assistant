import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer, DateTime, JSON, Boolean, ForeignKey
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    """
    One row per Slack user. Persists across days.
    A user always has exactly one active session; new uploads append to it.
    """
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    slack_user_id = Column(String(64), unique=True, nullable=False, index=True)
    slack_channel_id = Column(String(64), nullable=False)

    # Intake outputs
    transcript_raw = Column(Text, nullable=True)
    client_data = Column(JSON, nullable=True)          # Dict matching ClientData
    missing_fields = Column(JSON, nullable=True)       # List[str]
    intake_complete = Column(Boolean, default=False)

    # Research outputs
    retrieved_chunk_ids = Column(JSON, nullable=True)  # List[str] – vector ids
    research_complete = Column(Boolean, default=False)

    # Writer outputs
    proposal_sections = Column(JSON, nullable=True)    # Dict[section_name, text]
    docx_path = Column(String(512), nullable=True)
    writer_complete = Column(Boolean, default=False)

    # Revision tracking
    revision_count = Column(Integer, default=0)

    # Lifecycle
    status = Column(String(32), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship(
        "Message", back_populates="session",
        order_by="Message.created_at", cascade="all, delete-orphan"
    )
    revisions = relationship(
        "Revision", back_populates="session",
        order_by="Revision.created_at", cascade="all, delete-orphan"
    )


class Message(Base):
    """
    Full conversation history for a session (used to reconstruct LangGraph state).
    """
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)          # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


class Revision(Base):
    """
    Each time the user requests a change, we log the request, which section
    was targeted, and the resulting docx path.
    """
    __tablename__ = "revisions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False, index=True)
    revision_number = Column(Integer, nullable=False)
    request_text = Column(Text, nullable=False)
    target_section = Column(String(128), nullable=True)
    docx_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="revisions")
