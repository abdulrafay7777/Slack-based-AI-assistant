"""
Slack event handler (FastAPI router).

Handles:
  - file_shared events → transcript ingestion → full pipeline
  - message events     → follow-up Q&A or revision requests

Design decision: Slack event handling is synchronous from Slack's perspective
(we must respond within 3 seconds). Heavy work is kicked off as a background
task so we can acknowledge Slack immediately.
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Request, Response
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier

from app.db.session import AsyncSessionLocal
from app.db import crud
from app.agents.graph import proposal_graph, revision_graph
from app.state import ProposalState

router = APIRouter()

_slack_client = AsyncWebClient(token=os.getenv("SLACK_BOT_TOKEN"))
_verifier = SignatureVerifier(signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""))

# Helpers
async def _download_slack_file(file_url: str) -> str:
    """Download a Slack-hosted file and return its text content."""
    headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(file_url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text


async def _post_message(channel: str, text: str) -> None:
    await _slack_client.chat_postMessage(channel=channel, text=text)


async def _upload_docx(channel: str, docx_path: str, filename: str, comment: str) -> None:
    """Upload a DOCX file to Slack."""
    with open(docx_path, "rb") as f:
        await _slack_client.files_upload_v2(
            channel=channel,
            file=f,
            filename=filename,
            initial_comment=comment,
        )


def _detect_revision(text: str) -> tuple[bool, str | None]:
    """
    Detect whether a message is a revision request vs a question.

    Returns (is_revision, target_section_hint).
    Simple heuristic: revision keywords trigger revision mode.
    """
    revision_keywords = [
        "make", "update", "change", "revise", "rewrite", "expand",
        "shorten", "add", "remove", "adjust", "improve", "more detail",
        "less detail", "fix",
    ]
    lower = text.lower()
    is_revision = any(kw in lower for kw in revision_keywords)

    target = None
    sections = {
        "cover": ["cover", "title"],
        "executive_summary": ["executive", "summary", "exec summary"],
        "proposed_solution": ["solution", "approach", "proposed"],
        "timeline": ["timeline", "phases", "schedule"],
        "budget": ["budget", "price", "pricing", "cost"],
        "next_steps": ["next steps", "action"]
    }
    for sec, kws in sections.items():
        if any(kw in lower for kw in kws):
            target = sec
            break

    return is_revision, target

# Background tasks
async def _run_full_pipeline(
    session_id: str,
    slack_user_id: str,
    slack_channel_id: str,
    transcript: str,
) -> None:
    """Run intake → research → writer and deliver DOCX to Slack."""
    await _post_message(slack_channel_id, "⏳ Processing your transcript… this takes about 30 seconds.")

    async with AsyncSessionLocal() as db:
        # Append transcript to session (handles multi-upload)
        await crud.append_transcript(db, session_id, transcript)
        session = await crud.get_session_by_user(db, slack_user_id)
        messages = await crud.get_messages(db, session_id)

    # Build initial LangGraph state
    initial_state: ProposalState = {
        "session_id": session_id,
        "slack_user_id": slack_user_id,
        "slack_channel_id": slack_channel_id,
        "transcript_raw": session.transcript_raw,
        "client_data": None,
        "missing_fields": None,
        "intake_complete": False,
        "retrieved_chunks": None,
        "research_complete": False,
        "proposal_sections": None,
        "docx_path": None,
        "writer_complete": False,
        "revision_request": None,
        "target_section": None,
        "revision_count": session.revision_count,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "error": None,
        "status": "processing",
    }

    final_state = await proposal_graph.ainvoke(initial_state)

    if final_state.get("error"):
        await _post_message(slack_channel_id, f"❌ Error: {final_state['error']}")
        return

    # Persist results
    async with AsyncSessionLocal() as db:
        await crud.update_session(
            db,
            session_id,
            client_data=final_state.get("client_data"),
            missing_fields=final_state.get("missing_fields"),
            intake_complete=final_state.get("intake_complete", False),
            research_complete=final_state.get("research_complete", False),
            proposal_sections=final_state.get("proposal_sections"),
            docx_path=final_state.get("docx_path"),
            writer_complete=final_state.get("writer_complete", False),
            status=final_state.get("status", "done"),
        )
        await crud.add_message(db, session_id, "assistant",
                               "Proposal generated. See attached DOCX.")

    # Notify missing fields
    missing = final_state.get("missing_fields") or []
    if missing:
        await _post_message(
            slack_channel_id,
            f"⚠️ The transcript was missing information for: *{', '.join(missing)}*. "
            "I've done my best without them, but providing these will strengthen the proposal."
        )

    # Deliver DOCX
    docx_path = final_state.get("docx_path")
    if docx_path:
        await _upload_docx(
            channel=slack_channel_id,
            docx_path=docx_path,
            filename=f"proposal_{session_id[:8]}_draft.docx",
            comment="✅ Your draft proposal is ready! Reply to request changes, ask questions, or say 'approve' to finalize delivery.",
        )
    else:
        await _post_message(slack_channel_id, "⚠️ Proposal text generated but DOCX creation failed.")


async def _run_revision(
    session_id: str,
    slack_user_id: str,
    slack_channel_id: str,
    revision_request: str,
    target_section: str | None,
) -> None:
    """Run a targeted revision and deliver updated DOCX."""
    await _post_message(slack_channel_id, "✏️ Revising the proposal…")

    async with AsyncSessionLocal() as db:
        session = await crud.get_session_by_user(db, slack_user_id)
        if not session or not session.proposal_sections:
            await _post_message(slack_channel_id, "❌ No existing proposal found. Please upload a transcript first.")
            return
        revision_count = (session.revision_count or 0) + 1

    initial_state: ProposalState = {
        "session_id": session_id,
        "slack_user_id": slack_user_id,
        "slack_channel_id": slack_channel_id,
        "transcript_raw": session.transcript_raw,
        "client_data": session.client_data,
        "missing_fields": session.missing_fields,
        "intake_complete": True,
        "retrieved_chunks": None,
        "research_complete": True,
        "proposal_sections": session.proposal_sections,
        "docx_path": session.docx_path,
        "writer_complete": False,
        "revision_request": revision_request,
        "target_section": target_section,
        "revision_count": revision_count,
        "messages": [],
        "error": None,
        "status": "revising",
    }

    final_state = await revision_graph.ainvoke(initial_state)

    if final_state.get("error"):
        await _post_message(slack_channel_id, f"❌ Revision error: {final_state['error']}")
        return

    async with AsyncSessionLocal() as db:
        await crud.update_session(
            db, session_id,
            proposal_sections=final_state.get("proposal_sections"),
            docx_path=final_state.get("docx_path"),
            revision_count=revision_count,
            status="done",
        )
        await crud.add_revision(
            db,
            session_id=session_id,
            revision_number=revision_count,
            request_text=revision_request,
            target_section=target_section,
            docx_path=final_state.get("docx_path"),
        )

    docx_path = final_state.get("docx_path")
    if docx_path:
        await _upload_docx(
            channel=slack_channel_id,
            docx_path=docx_path,
            filename=f"proposal_{session_id[:8]}_rev{revision_count}.docx",
            comment=f"✅ Revision {revision_count} complete! Section updated: *{target_section or 'inferred from request'}*.",
        )


async def _answer_question(
    session_id: str,
    slack_channel_id: str,
    question: str,
    slack_user_id: str,
) -> None:
    """Answer a follow-up question using session context."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    async with AsyncSessionLocal() as db:
        session = await crud.get_session_by_user(db, slack_user_id)
        await crud.add_message(db, session_id, "user", question)

    context_parts = []
    if session and session.transcript_raw:
        context_parts.append(f"TRANSCRIPT:\n{session.transcript_raw[:3000]}")
    if session and session.client_data:
        import json
        context_parts.append(f"CLIENT DATA:\n{json.dumps(session.client_data, indent=2)}")
    if session and session.proposal_sections:
        import json
        context_parts.append(f"PROPOSAL SECTIONS:\n{json.dumps(session.proposal_sections, indent=2)}")

    context = "\n\n".join(context_parts) or "No session context available."

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    response = await llm.ainvoke([
        SystemMessage(content=(
            "You are a helpful assistant for a business consultant. "
            "Answer questions about the client transcript and proposal using the context below. "
            "Be concise and specific. If the answer is not in the context, say so."
        )),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ])

    answer = response.content.strip()

    async with AsyncSessionLocal() as db:
        await crud.add_message(db, session_id, "assistant", answer)

    await _post_message(slack_channel_id, answer)

# Main event endpoint
@router.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks) -> Response:
    # Read body once and parse
    body_bytes = await request.body()
    body = None
    try:
        import json
        body = json.loads(body_bytes.decode())
    except Exception:
        return Response(status_code=400, content="Invalid JSON")

    # URL verification challenge (Slack sends this once during app setup)
    # MUST be handled BEFORE signature verification
    if body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        return Response(content=challenge, media_type="text/plain")

    # Verify signature for all other events
    if os.getenv("VERIFY_SLACK_SIGNATURE", "true").lower() == "true":
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")
        if not _verifier.is_valid(body=body_bytes.decode(), timestamp=timestamp, signature=signature):
            return Response(status_code=403)

    event = body.get("event", {})
    event_type = event.get("type")
    user_id = event.get("user")
    channel_id = event.get("channel")

    # Ignore bot's own messages
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return Response(status_code=200)

    # File upload → trigger full pipeline
    if event_type == "message" and event.get("files"):
        files = event["files"]
        txt_files = [f for f in files if f.get("filetype") in ("text", "txt", "plain")]
        if not txt_files:
            background_tasks.add_task(
                _post_message, channel_id,
                "Please upload a plain text (.txt) transcript file."
            )
            return Response(status_code=200)

        async def _handle_upload():
            async with AsyncSessionLocal() as db:
                session, created = await crud.get_or_create_session(db, user_id, channel_id)
            if created:
                await _post_message(channel_id, f"👋 New session started for <@{user_id}>!")

            for file_info in txt_files:
                url = file_info.get("url_private_download") or file_info.get("url_private")
                transcript = await _download_slack_file(url)
                await _run_full_pipeline(session.id, user_id, channel_id, transcript)

        background_tasks.add_task(_handle_upload)
        return Response(status_code=200)

    # Text message → Q&A or revision
    if event_type == "message" and event.get("text"):
        text = event["text"].strip()

        # Only respond to DMs or @mentions
        bot_user_id = os.getenv("SLACK_BOT_USER_ID", "")
        is_dm = event.get("channel_type") == "im"
        is_mention = f"<@{bot_user_id}>" in text
        if not is_dm and not is_mention:
            return Response(status_code=200)

        # Strip mention from text
        text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        if not text:
            return Response(status_code=200)

        async with AsyncSessionLocal() as db:
            session, created = await crud.get_or_create_session(db, user_id, channel_id)
            if created or not session.transcript_raw:
                await _post_message(
                    channel_id,
                    "No session found yet. Please upload a transcript file to get started!"
                )
                return Response(status_code=200)

        is_revision, target_section = _detect_revision(text)

        if text.lower() == "approve":
            background_tasks.add_task(
                _post_message, channel_id, "🎉 Proposal approved! The final version is ready for client delivery."
            )
        elif is_revision and session.proposal_sections:
            background_tasks.add_task(
                _run_revision,
                session.id, user_id, channel_id,
                text, target_section,
            )
        else:
            background_tasks.add_task(
                _answer_question,
                session.id, channel_id, text, user_id,
            )

    return Response(status_code=200)
