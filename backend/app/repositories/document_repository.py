import uuid

from sqlalchemy import ColumnElement, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import (
    Document,
    DocumentChunk,
    DocumentPermission,
    DocumentVersion,
    IngestionJob,
)
from app.models.user import ROLE_ADMIN, User


def accessible_documents_clause(user: User) -> ColumnElement[bool]:
    """SQLAlchemy boolean expression, usable in any .where() alongside a query
    that has `Document` in its FROM/JOIN list: true iff `user` may see that row.

    Admins bypass entirely. Everyone else needs to own the document or have an
    explicit DocumentPermission grant (to them directly, or to their role) —
    the access-control principle from docs/ARCHITECTURE.md §10 risk 10.4: this
    is a join condition baked into the query, not a filter applied to results
    fetched from an unfiltered query. Shared between this repository and
    retrieval_service.py so the RAG query path and the document-browsing path
    can never drift apart on what counts as "accessible".
    """
    if user.role.name == ROLE_ADMIN:
        return true()

    return or_(
        Document.owner_id == user.id,
        select(DocumentPermission.id)
        .where(
            DocumentPermission.document_id == Document.id,
            or_(
                DocumentPermission.user_id == user.id,
                DocumentPermission.role_id == user.role_id,
            ),
        )
        .exists(),
    )


async def list_documents(session: AsyncSession, user: User) -> list[Document]:
    result = await session.execute(
        select(Document)
        .where(accessible_documents_clause(user))
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def get_document(
    session: AsyncSession, document_id: uuid.UUID, user: User
) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.id == document_id, accessible_documents_clause(user))
    )
    return result.scalar_one_or_none()


async def get_latest_ingestion_job(
    session: AsyncSession, document_id: uuid.UUID
) -> tuple[IngestionJob, int] | None:
    """Latest job by creation time, not `started_at` — a still-queued job has
    `started_at=NULL`, so sorting by that instead of `created_at` would bury a
    brand-new pending job under older, already-finished ones.
    """
    result = await session.execute(
        select(IngestionJob, DocumentVersion.version_number)
        .join(DocumentVersion, IngestionJob.document_version_id == DocumentVersion.id)
        .where(DocumentVersion.document_id == document_id)
        .order_by(IngestionJob.created_at.desc())
        .limit(1)
    )
    return result.tuples().first()


async def list_versions(session: AsyncSession, document_id: uuid.UUID) -> list[DocumentVersion]:
    result = await session.execute(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version_number.desc())
    )
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, document: Document) -> None:
    await session.delete(document)
    await session.commit()


async def get_chunk_with_document(
    session: AsyncSession, document_id: uuid.UUID, chunk_id: uuid.UUID, user: User
) -> tuple[DocumentChunk, DocumentVersion, Document] | None:
    result = await session.execute(
        select(DocumentChunk, DocumentVersion, Document)
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(
            DocumentChunk.id == chunk_id,
            Document.id == document_id,
            accessible_documents_clause(user),
        )
    )
    return result.tuples().first()
