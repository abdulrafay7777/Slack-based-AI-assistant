"""
All database read/write logic lives here.
Agents never import SQLAlchemy directly they call these functions.
"""
from __future__ import annotations

from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, Message, Revision


# Session
async def get_session_by_user(db: AsyncSession, slack_user_id: str) -> Optional[Session]:
    result = await db.execute(
        select(Session).where(Session.slack_user_id == slack_user_id)
    )
    return result.scalars().first()


async def create_session(
    db: AsyncSession,
    slack_user_id: str,
    slack_channel_id: str,
) -> Session:
    session = Session(
        slack_user_id=slack_user_id,
        slack_channel_id=slack_channel_id,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_or_create_session(
    db: AsyncSession,
    slack_user_id: str,
    slack_channel_id: str,
) -> tuple[Session, bool]:
    """Returns (session, created). created=True means this is a brand-new session."""
    existing = await get_session_by_user(db, slack_user_id)
    if existing:
        return existing, False
    new_session = await create_session(db, slack_user_id, slack_channel_id)
    return new_session, True


async def update_session(db: AsyncSession, session_id: str, **kwargs) -> Optional[Session]:
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalars().first()
    if not session:
        return None
    for key, value in kwargs.items():
        if hasattr(session, key):
            setattr(session, key, value)
    await db.commit()
    await db.refresh(session)
    return session


async def append_transcript(db: AsyncSession, session_id: str, new_text: str) -> Optional[Session]:
    """New uploads are appended to the existing transcript, not a fresh start."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalars().first()
    if not session:
        return None
    existing = session.transcript_raw or ""
    separator = "\n\n--- NEW UPLOAD ---\n\n" if existing else ""
    session.transcript_raw = existing + separator + new_text
    # Reset downstream state so agents re-run on new content
    session.intake_complete = False
    session.research_complete = False
    session.writer_complete = False
    session.status = "pending"
    await db.commit()
    await db.refresh(session)
    return session


# Messages
async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
) -> Message:
    msg = Message(session_id=session_id, role=role, content=content)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_messages(db: AsyncSession, session_id: str) -> List[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    return list(result.scalars().all())


# Revisions
async def add_revision(
    db: AsyncSession,
    session_id: str,
    revision_number: int,
    request_text: str,
    target_section: Optional[str],
    docx_path: Optional[str],
) -> Revision:
    rev = Revision(
        session_id=session_id,
        revision_number=revision_number,
        request_text=request_text,
        target_section=target_section,
        docx_path=docx_path,
    )
    db.add(rev)
    await db.commit()
    await db.refresh(rev)
    return rev


async def get_revisions(db: AsyncSession, session_id: str) -> List[Revision]:
    result = await db.execute(
        select(Revision)
        .where(Revision.session_id == session_id)
        .order_by(Revision.created_at)
    )
    return list(result.scalars().all())
