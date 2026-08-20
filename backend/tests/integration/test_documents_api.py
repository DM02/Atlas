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
from app.models.document import DocumentChunk, DocumentVersion
from tests.integration.factories import ImmediateArqPool


class FakeEmbeddingProvider:
    model_name = "fake-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def _wire_overrides(db_session, tmp_path) -> Settings:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

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


def _auth_header(user) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id=user.id)}"}


async def test_upload_list_status_and_delete_document(db_session, tmp_path, admin_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"Atlas is a RAG platform. " * 20, "text/plain")},
                headers=headers,
            )
            assert upload.status_code == 201
            body = upload.json()
            assert body["status"] == "ready"
            document_id = body["id"]

            listing = await client.get("/api/v1/documents", headers=headers)
            assert listing.status_code == 200
            assert any(d["id"] == document_id for d in listing.json())

            status_resp = await client.get(
                f"/api/v1/documents/{document_id}/status", headers=headers
            )
            assert status_resp.status_code == 200
            assert status_resp.json()["status"] == "ready"

            deleted = await client.delete(f"/api/v1/documents/{document_id}", headers=headers)
            assert deleted.status_code == 204

            missing = await client.get(f"/api/v1/documents/{document_id}", headers=headers)
            assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_upload_rejects_unsupported_extension(db_session, tmp_path, admin_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("malware.exe", b"x", "application/octet-stream")},
                headers=_auth_header(admin_user),
            )
            assert response.status_code == 415
    finally:
        app.dependency_overrides.clear()


async def test_upload_exposes_ingestion_failure_via_status_endpoint(
    db_session, tmp_path, admin_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
                headers=headers,
            )
            assert upload.status_code == 201
            document_id = upload.json()["id"]
            assert upload.json()["status"] == "failed"

            status_resp = await client.get(
                f"/api/v1/documents/{document_id}/status", headers=headers
            )
            assert status_resp.status_code == 200
            body = status_resp.json()
            assert body["status"] == "failed"
            assert body["error_message"]
    finally:
        app.dependency_overrides.clear()


async def test_get_chunk_returns_source_fragment(db_session, tmp_path, admin_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"Atlas is a RAG platform. " * 20, "text/plain")},
                headers=headers,
            )
            document_id = upload.json()["id"]

            version = (
                await db_session.execute(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == document_id
                    )
                )
            ).scalar_one()
            chunk = (
                await db_session.execute(
                    select(DocumentChunk).where(
                        DocumentChunk.document_version_id == version.id
                    )
                )
            ).scalars().first()

            response = await client.get(
                f"/api/v1/documents/{document_id}/chunks/{chunk.id}", headers=headers
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["document_title"] == "notes.txt"
            assert payload["content"]

            other_document_id = "00000000-0000-0000-0000-000000000000"
            wrong_doc = await client.get(
                f"/api/v1/documents/{other_document_id}/chunks/{chunk.id}", headers=headers
            )
            assert wrong_doc.status_code == 404
    finally:
        app.dependency_overrides.clear()
