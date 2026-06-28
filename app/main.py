"""
FastAPI application entry point.

Startup sequence:
  1. Init database (create tables if not exists)
  2. Ingest proposals into vector DB (if collection is empty)
  3. Register Slack event router
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
load_dotenv() 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import init_db
from app.vector.retriever import collection_count
from app.api.slack_events import router as slack_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic using the modern lifespan pattern."""
    # --- Startup ---
    logger.info("Initialising database...")
    await init_db()
    logger.info("Database ready.")

    count = await collection_count()
    if count == 0:
        logger.info("Vector collection is empty - running proposal ingestion...")
        from scripts.ingest_proposals import ingest_all
        await ingest_all()
        logger.info("Ingestion complete.")
    else:
        logger.info(f"Vector collection has {count} chunks - skipping ingestion.")

    yield  # App is running

    # --- Shutdown (add cleanup here if needed) ---
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Proposal Assistant",
        description="Slack-based AI assistant for generating business proposals",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(slack_router)

    @app.get("/")
    async def root() -> dict:
        return {
            "service": "Proposal Assistant",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "slack_events": "/slack/events"
            }
        }

    @app.post("/")
    async def root_post() -> dict:
        return {
            "message": "Please use /slack/events for Slack event webhooks",
            "endpoint": "/slack/events"
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENV", "production") == "development",
    )
