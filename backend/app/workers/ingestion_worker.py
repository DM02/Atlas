import uuid
from typing import Any

from arq.connections import RedisSettings

from app.ai.embeddings.factory import get_embedding_provider
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.storage import get_storage_backend
from app.db.session import async_session_factory
from app.services.ingestion_service import process_document_version


async def _on_startup(ctx: dict[str, Any]) -> None:
    # Runs in the worker's own process (separate from the FastAPI app, which
    # configures logging in main.py at import time) — structlog's config is
    # per-process, so each process needs its own call.
    configure_logging()


async def run_ingestion(ctx: dict[str, Any], document_version_id: str) -> None:
    """Arq task: runs the actual document pipeline for a version that
    create_pending_version already created (status=pending). Opens its own DB
    session — this runs in a separate worker process from the FastAPI app, so
    it can't reuse a request-scoped session.
    """
    settings = get_settings()
    async with async_session_factory() as session:
        await process_document_version(
            session=session,
            storage=get_storage_backend(),
            embedding_provider=get_embedding_provider(),
            settings=settings,
            document_version_id=uuid.UUID(document_version_id),
        )


class WorkerSettings:
    functions = [run_ingestion]
    on_startup = _on_startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
