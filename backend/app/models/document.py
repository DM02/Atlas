import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.config import get_settings
from app.db.base import Base

settings = get_settings()

# Document processing status, shared by Document and IngestionJob.
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default=STATUS_PENDING)
    # The version currently served to readers (retrieval only ever searches this
    # one, see retrieval_service.py). Set only once a version's processing
    # succeeds — a version that fails to process never becomes current, so a
    # document with a previously-good version keeps serving it instead of going
    # dark on a bad re-upload. NULL until the first version ever succeeds.
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )
    permissions: Mapped[list["DocumentPermission"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version_number"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    storage_path: Mapped[str] = mapped_column(String(1024))
    checksum: Mapped[str] = mapped_column(String(64))
    # Each version's own type — a re-upload can be a different file type than the
    # version it replaces (e.g. .txt then .pdf). Document.mime_type mirrors
    # whichever version is current, but extraction must always use *this*
    # column, not the document's — see process_document_version.
    mime_type: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped["Document"] = relationship(
        back_populates="versions", foreign_keys="DocumentVersion.document_id"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )
    ingestion_jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.openai_embedding_dimension))
    # Postgres-computed (GENERATED ALWAYS AS ... STORED, see the migration). `Computed()`
    # tells SQLAlchemy never to send this column in INSERT/UPDATE — without it, the ORM
    # sends an implicit NULL for any mapped column you didn't set, which Postgres rejects
    # for a generated column. Exists on the model so SQLAlchemy can build full-text search
    # expressions against it (see retrieval_service._fts_candidates); never set from Python.
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True), nullable=True
    )

    document_version: Mapped["DocumentVersion"] = relationship(back_populates="chunks")


class DocumentPermission(Base):
    """Grants a user or a role read access to a document. Exactly one of
    user_id/role_id is normally set — enforced loosely (at least one, not
    exactly one) since a redundant double-grant is harmless, just unusual.
    Retrieval queries join against this table directly; see
    retrieval_service.py and docs/ARCHITECTURE.md §10 risk 10.4.
    """

    __tablename__ = "document_permissions"
    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL) OR (role_id IS NOT NULL)",
            name="ck_document_permission_target",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    role_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("roles.id", ondelete="CASCADE"), nullable=True
    )
    permission_level: Mapped[str] = mapped_column(String(16), default="read")

    document: Mapped["Document"] = relationship(back_populates="permissions")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(32), default=STATUS_PENDING)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    document_version: Mapped["DocumentVersion"] = relationship(back_populates="ingestion_jobs")
