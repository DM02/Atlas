import hashlib
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.pipeline.chunking import chunk_pages
from app.ai.pipeline.extraction import EXTENSION_TO_MIME, extract_text
from app.core.config import Settings
from app.core.storage import StorageBackend
from app.models.document import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_READY,
    Document,
    DocumentChunk,
    DocumentPermission,
    DocumentVersion,
    IngestionJob,
)
from app.models.metrics import RequestMetric
from app.models.user import ROLE_USER, Role

logger = structlog.get_logger(__name__)


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


def _validate(filename: str, content: bytes, settings: Settings) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in EXTENSION_TO_MIME or extension not in settings.allowed_upload_extensions:
        raise UnsupportedFileTypeError(f"Unsupported file extension: {extension or '(none)'}")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeError(f"File exceeds max size of {settings.max_upload_size_mb}MB")

    return extension


async def create_pending_version(
    *,
    session: AsyncSession,
    storage: StorageBackend,
    settings: Settings,
    filename: str,
    content: bytes,
    owner_id: uuid.UUID,
    document: Document | None = None,
    is_private: bool = False,
) -> DocumentVersion:
    """Validates and stores an upload and creates the tracking rows (Document if
    new, DocumentVersion, IngestionJob — all left at status=pending) without
    running the pipeline. Fast enough to run inline in an HTTP request; the
    actual extract/chunk/embed work happens in process_document_version, which
    the API enqueues onto the Arq worker (see workers/ingestion_worker.py) so
    the request doesn't block on it.

    Pass an existing `document` to add a new version to it (re-upload) — the
    document's own visibility (`is_private`) isn't touched by a re-upload, only
    by first creation. Validation failures (bad extension, oversized file) raise
    before any DB row is created, since those are the caller's fault, not a
    processing outcome.
    """
    extension = _validate(filename, content, settings)
    mime_type = EXTENSION_TO_MIME[extension]
    checksum = hashlib.sha256(content).hexdigest()

    if document is None:
        document = Document(owner_id=owner_id, title=filename, mime_type=mime_type)
        session.add(document)
        await session.flush()

        if not is_private:
            user_role_id = (
                await session.execute(select(Role.id).where(Role.name == ROLE_USER))
            ).scalar_one()
            session.add(DocumentPermission(document_id=document.id, role_id=user_role_id))

    next_version_number = (
        await session.execute(
            select(func.coalesce(func.max(DocumentVersion.version_number), 0)).where(
                DocumentVersion.document_id == document.id
            )
        )
    ).scalar_one() + 1

    # storage_path stores the backend's content-addressing KEY, not a resolved
    # filesystem path — process_document_version reads it back from a different
    # process (the worker), so it must be something storage.read() can resolve
    # on its own, not a path that only meant something in this request.
    storage_key = f"{document.id}/v{next_version_number}{extension}"
    await storage.save(storage_key, content)

    version = DocumentVersion(
        document_id=document.id,
        version_number=next_version_number,
        storage_path=storage_key,
        checksum=checksum,
        mime_type=mime_type,
    )
    session.add(version)
    await session.flush()

    session.add(IngestionJob(document_version_id=version.id, status=STATUS_PENDING))

    await session.commit()
    return version


async def process_document_version(
    *,
    session: AsyncSession,
    storage: StorageBackend,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    document_version_id: uuid.UUID,
) -> None:
    """Runs the pipeline (extract, clean, chunk, embed, index) for an
    already-created DocumentVersion. Called by the Arq worker in production,
    and directly (synchronously) in tests that want the old one-shot behavior.

    On success, the document's current_version_id cuts over to this version —
    that's the only thing that makes a version "live" for retrieval. On
    failure, a document that already had a good current_version_id keeps
    serving it (status stays ready); only a document with no prior good
    version goes to failed. A bad re-upload should degrade gracefully, not
    take down a document that was working.
    """
    version = await session.get(DocumentVersion, document_version_id)
    if version is None:
        raise ValueError(f"DocumentVersion {document_version_id} does not exist")
    document = await session.get(Document, version.document_id)
    if document is None:
        raise ValueError(f"Document {version.document_id} does not exist")

    job = (
        await session.execute(
            select(IngestionJob)
            .where(IngestionJob.document_version_id == version.id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
    ).scalars().first()
    if job is None:
        raise ValueError(f"No IngestionJob found for DocumentVersion {document_version_id}")

    job.status = STATUS_PROCESSING
    job.started_at = datetime.now(UTC)
    document.status = STATUS_PROCESSING
    await session.commit()

    structlog.contextvars.bind_contextvars(document_version_id=str(version.id))
    pipeline_start = time.perf_counter()

    try:
        extract_start = time.perf_counter()
        content = await storage.read(version.storage_path)
        pages = extract_text(content, version.mime_type)
        extract_ms = (time.perf_counter() - extract_start) * 1000

        chunk_start = time.perf_counter()
        chunks = chunk_pages(
            pages,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        chunk_ms = (time.perf_counter() - chunk_start) * 1000

        embed_ms = 0.0
        if chunks:
            embed_start = time.perf_counter()
            embeddings = await embedding_provider.embed_documents([c.content for c in chunks])
            embed_ms = (time.perf_counter() - embed_start) * 1000
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                session.add(
                    DocumentChunk(
                        document_version_id=version.id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        token_count=chunk.token_count,
                        embedding_model=embedding_provider.model_name,
                        embedding=embedding,
                    )
                )

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "document_ingestion_completed",
            extract_ms=round(extract_ms, 1),
            chunk_ms=round(chunk_ms, 1),
            embed_ms=round(embed_ms, 1),
            total_ms=round(total_ms, 1),
            num_chunks=len(chunks),
        )
        session.add(
            RequestMetric(
                endpoint="document_ingestion",
                stage_latencies_ms={
                    "extract_ms": round(extract_ms, 1),
                    "chunk_ms": round(chunk_ms, 1),
                    "embed_ms": round(embed_ms, 1),
                },
                total_ms=round(total_ms),
            )
        )

        job.status = STATUS_READY
        job.finished_at = datetime.now(UTC)
        document.status = STATUS_READY
        document.current_version_id = version.id
        document.mime_type = version.mime_type
    except Exception as exc:
        job.status = STATUS_FAILED
        job.error_message = str(exc)[:2000]
        job.finished_at = datetime.now(UTC)
        document.status = STATUS_READY if document.current_version_id else STATUS_FAILED
    finally:
        await session.commit()
