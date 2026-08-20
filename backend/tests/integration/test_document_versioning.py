"""Phase 4 exit criterion: re-uploading a document creates a new version
without breaking old citations. Covers the service-level cutover logic
(current_version_id, graceful degradation on a failed re-upload) and the API
surface (PUT for re-upload, GET .../versions for history).
"""

from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.ai.embeddings.factory import get_embedding_provider
from app.core.config import Settings, get_settings
from app.core.queue import get_arq_pool
from app.core.security import create_access_token
from app.core.storage import LocalStorageBackend, get_storage_backend
from app.db.session import get_db
from app.main import app
from app.models.document import STATUS_FAILED, STATUS_READY, DocumentChunk, DocumentVersion
from app.services.ingestion_service import create_pending_version, process_document_version
from app.services.retrieval_service import retrieve_relevant_chunks
from tests.integration.factories import ImmediateArqPool, ingest_test_document


class KeywordEmbeddingProvider:
    model_name = "fake-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def _vector_for(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        lowered = text.lower()
        if "apple" in lowered:
            vec[0] = 1.0
        if "orange" in lowered:
            vec[1] = 1.0
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)


def _auth_header(user) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id=user.id)}"}


def _wire_overrides(db_session, tmp_path, embeddings) -> Settings:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))

    async def override_get_db() -> AsyncGenerator:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_storage_backend] = lambda: storage
    app.dependency_overrides[get_embedding_provider] = lambda: embeddings
    app.dependency_overrides[get_arq_pool] = lambda: ImmediateArqPool(
        session=db_session, storage=storage, embedding_provider=embeddings, settings=settings
    )
    return settings


# --- service level -----------------------------------------------------


async def test_reupload_creates_new_version_and_cuts_over_current(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"apple " * 20,
        owner_id=admin_user.id,
    )
    v1_id = document.current_version_id

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"orange " * 20,
        owner_id=admin_user.id,
        document=document,
    )

    versions = (
        (
            await db_session.execute(
                select(DocumentVersion)
                .where(DocumentVersion.document_id == document.id)
                .order_by(DocumentVersion.version_number)
            )
        )
        .scalars()
        .all()
    )
    assert [v.version_number for v in versions] == [1, 2]
    assert document.current_version_id == versions[1].id
    assert document.current_version_id != v1_id


async def test_retrieval_only_sees_current_version_not_old(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"apple " * 20,
        owner_id=admin_user.id,
    )
    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"orange " * 20,
        owner_id=admin_user.id,
        document=document,
    )

    # There's no similarity threshold (retrieval always returns its best available
    # candidates, however weak the match — reranking/the LLM decide relevance
    # downstream), so a query for "apple" still returns *a* result once v2 exists.
    # What matters here is that it can no longer be v1's "apple" chunk.
    apple_results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=5,
        requesting_user=admin_user,
    )
    assert all("apple" not in r.content for r in apple_results)

    orange_results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="orange",
        top_k=5,
        requesting_user=admin_user,
    )
    assert len(orange_results) == 1
    assert "orange" in orange_results[0].content


async def test_failed_reupload_does_not_regress_a_working_document(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"apple " * 20,
        owner_id=admin_user.id,
    )
    assert document.status == STATUS_READY
    good_version_id = document.current_version_id

    # Re-upload with content that will fail extraction (claims to be a PDF, isn't).
    # Same bytes as test_ingest_document_marks_failed_on_extraction_error — pypdf's
    # behavior on garbage input isn't consistent (some raise, some silently parse as
    # zero pages), so stick to bytes already proven to raise.
    bad_version = await create_pending_version(
        session=db_session,
        storage=storage,
        settings=settings,
        filename="broken.pdf",
        content=b"this is not a real pdf file",
        owner_id=admin_user.id,
        document=document,
    )
    await process_document_version(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        document_version_id=bad_version.id,
    )

    await db_session.refresh(document)
    assert document.status == STATUS_READY  # still serving the old good version
    assert document.current_version_id == good_version_id  # not cut over to the bad one

    # And retrieval still finds the original content.
    results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=5,
        requesting_user=admin_user,
    )
    assert len(results) == 1


async def test_failed_first_version_leaves_document_failed(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    document = await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="broken.pdf",
        content=b"not a real pdf",
        owner_id=admin_user.id,
    )

    assert document.status == STATUS_FAILED
    assert document.current_version_id is None


# --- API level -----------------------------------------------------


async def test_old_citation_still_resolves_after_a_new_version_is_uploaded(
    db_session, tmp_path, admin_user
) -> None:
    embeddings = KeywordEmbeddingProvider(1536)
    _wire_overrides(db_session, tmp_path, embeddings)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"apple " * 20, "text/plain")},
                headers=headers,
            )
            document_id = upload.json()["id"]

            v1 = (
                await db_session.execute(
                    select(DocumentVersion).where(DocumentVersion.document_id == document_id)
                )
            ).scalar_one()
            old_chunk = (
                (
                    await db_session.execute(
                        select(DocumentChunk).where(
                            DocumentChunk.document_version_id == v1.id
                        )
                    )
                )
                .scalars()
                .first()
            )

            # Re-upload — a new version supersedes v1.
            reupload = await client.put(
                f"/api/v1/documents/{document_id}",
                files={"file": ("notes.txt", b"orange " * 20, "text/plain")},
                headers=headers,
            )
            assert reupload.status_code == 202
            assert reupload.json()["version_number"] == 2
            assert reupload.json()["is_current"] is True  # ImmediateArqPool ran it inline

            # The old chunk is no longer part of the current version...
            document = await client.get(f"/api/v1/documents/{document_id}", headers=headers)
            assert document.json()["status"] == "ready"

            # ...but its citation/chunk is still fetchable by id — nothing was deleted.
            old_chunk_resp = await client.get(
                f"/api/v1/documents/{document_id}/chunks/{old_chunk.id}", headers=headers
            )
            assert old_chunk_resp.status_code == 200
            assert "apple" in old_chunk_resp.json()["content"]
    finally:
        app.dependency_overrides.clear()


async def test_get_versions_lists_history_with_correct_current_flag(
    db_session, tmp_path, admin_user
) -> None:
    embeddings = KeywordEmbeddingProvider(1536)
    _wire_overrides(db_session, tmp_path, embeddings)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"apple " * 20, "text/plain")},
                headers=headers,
            )
            document_id = upload.json()["id"]

            await client.put(
                f"/api/v1/documents/{document_id}",
                files={"file": ("notes.txt", b"orange " * 20, "text/plain")},
                headers=headers,
            )

            versions = await client.get(
                f"/api/v1/documents/{document_id}/versions", headers=headers
            )
            assert versions.status_code == 200
            body = versions.json()
            assert [v["version_number"] for v in body] == [2, 1]  # newest first
            assert body[0]["is_current"] is True
            assert body[1]["is_current"] is False
    finally:
        app.dependency_overrides.clear()
