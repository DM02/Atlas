import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.core.storage import LocalStorageBackend
from app.models.document import STATUS_FAILED, STATUS_READY, DocumentChunk, DocumentVersion
from app.services.ingestion_service import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    create_pending_version,
)
from tests.integration.factories import ingest_test_document


class FakeEmbeddingProvider:
    model_name = "fake-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


async def test_ingest_document_creates_ready_document_with_chunks(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"Atlas is a RAG platform. " * 50,
        owner_id=admin_user.id,
    )

    assert document.status == STATUS_READY
    assert document.owner_id == admin_user.id
    assert document.current_version_id is not None

    version = (
        await db_session.execute(
            select(DocumentVersion).where(DocumentVersion.document_id == document.id)
        )
    ).scalar_one()
    assert document.current_version_id == version.id

    chunks = (
        (
            await db_session.execute(
                select(DocumentChunk).where(DocumentChunk.document_version_id == version.id)
            )
        )
        .scalars()
        .all()
    )

    assert len(chunks) >= 1
    assert chunks[0].embedding_model == "fake-embed"
    assert chunks[0].token_count > 0


async def test_create_pending_version_rejects_unsupported_extension(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))

    with pytest.raises(UnsupportedFileTypeError):
        await create_pending_version(
            session=db_session,
            storage=storage,
            settings=settings,
            filename="malware.exe",
            content=b"whatever",
            owner_id=admin_user.id,
        )


async def test_create_pending_version_rejects_oversized_file(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path), max_upload_size_mb=0)
    storage = LocalStorageBackend(str(tmp_path))

    with pytest.raises(FileTooLargeError):
        await create_pending_version(
            session=db_session,
            storage=storage,
            settings=settings,
            filename="notes.txt",
            content=b"some content",
            owner_id=admin_user.id,
        )


async def test_ingest_document_marks_failed_on_extraction_error(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="broken.pdf",
        content=b"this is not a real pdf file",
        owner_id=admin_user.id,
    )

    assert document.status == STATUS_FAILED
    assert document.current_version_id is None
