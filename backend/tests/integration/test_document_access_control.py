"""Phase 3 exit criterion: an unauthorized user must provably not be able to
retrieve a restricted chunk — via the RAG retrieval path (retrieval_service),
the direct chunk-inspection endpoint, the document list/detail endpoints, and
a full chat query. All four are tested here because "provably cannot" means
every door, not just the obvious one.
"""

from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.ai.embeddings.factory import get_embedding_provider
from app.ai.llm.factory import get_llm_provider
from app.core.config import Settings, get_settings
from app.core.queue import get_arq_pool
from app.core.security import create_access_token
from app.core.storage import LocalStorageBackend, get_storage_backend
from app.db.session import get_db
from app.main import app
from app.models.document import DocumentChunk, DocumentVersion
from app.services.retrieval_service import retrieve_relevant_chunks
from tests.integration.factories import ImmediateArqPool, ingest_test_document


class FakeEmbeddingProvider:
    model_name = "fake-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * self.dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class StubLLM:
    model_name = "stub"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "Answer based on context [1]."


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
    app.dependency_overrides[get_llm_provider] = lambda: StubLLM()
    app.dependency_overrides[get_arq_pool] = lambda: ImmediateArqPool(
        session=db_session, storage=storage, embedding_provider=embeddings, settings=settings
    )
    return settings


def _auth_header(user) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id=user.id)}"}


# --- retrieval_service level (no HTTP) -------------------------------------


async def test_private_document_excluded_from_retrieval_for_unauthorized_user(
    db_session, tmp_path, admin_user, regular_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="secret.txt",
        content=b"The launch codes are hidden here. " * 10,
        owner_id=admin_user.id,
        is_private=True,
    )

    as_stranger = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="launch codes",
        top_k=5,
        requesting_user=regular_user,
    )
    assert as_stranger == []

    as_owner = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="launch codes",
        top_k=5,
        requesting_user=admin_user,
    )
    assert len(as_owner) == 1
    assert as_owner[0].document_title == "secret.txt"


async def test_default_visibility_grants_access_to_any_regular_user(
    db_session, tmp_path, admin_user, regular_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="handbook.txt",
        content=b"Company holidays are listed here. " * 10,
        owner_id=admin_user.id,
        is_private=False,
    )

    results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="holidays",
        top_k=5,
        requesting_user=regular_user,
    )
    assert len(results) == 1
    assert results[0].document_title == "handbook.txt"


# --- API level (HTTP, real JWTs) --------------------------------------------


async def test_unauthorized_user_cannot_fetch_a_restricted_chunk_via_api(
    db_session, tmp_path, admin_user, regular_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                data={"private": "true"},
                files={"file": ("secret.txt", b"classified content here. " * 10, "text/plain")},
                headers=_auth_header(admin_user),
            )
            assert upload.status_code == 201
            document_id = upload.json()["id"]

            version = (
                await db_session.execute(
                    select(DocumentVersion).where(DocumentVersion.document_id == document_id)
                )
            ).scalar_one()
            chunk = (
                (
                    await db_session.execute(
                        select(DocumentChunk).where(
                            DocumentChunk.document_version_id == version.id
                        )
                    )
                )
                .scalars()
                .first()
            )

            as_owner = await client.get(
                f"/api/v1/documents/{document_id}/chunks/{chunk.id}",
                headers=_auth_header(admin_user),
            )
            assert as_owner.status_code == 200

            as_stranger = await client.get(
                f"/api/v1/documents/{document_id}/chunks/{chunk.id}",
                headers=_auth_header(regular_user),
            )
            assert as_stranger.status_code == 404

            listing = await client.get("/api/v1/documents", headers=_auth_header(regular_user))
            assert document_id not in [d["id"] for d in listing.json()]

            detail = await client.get(
                f"/api/v1/documents/{document_id}", headers=_auth_header(regular_user)
            )
            assert detail.status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_non_admin_cannot_upload_documents(db_session, tmp_path, regular_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"hello", "text/plain")},
                headers=_auth_header(regular_user),
            )
            assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()


async def test_document_endpoints_reject_missing_or_invalid_token(
    db_session, tmp_path
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            no_token = await client.get("/api/v1/documents")
            assert no_token.status_code == 401

            bad_token = await client.get(
                "/api/v1/documents", headers={"Authorization": "Bearer not-a-real-token"}
            )
            assert bad_token.status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_chat_query_never_cites_a_restricted_document_for_unauthorized_user(
    db_session, tmp_path, admin_user, regular_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/documents",
                data={"private": "true"},
                files={"file": ("secret.txt", b"classified content here. " * 10, "text/plain")},
                headers=_auth_header(admin_user),
            )

            response = await client.post(
                "/api/v1/chat/query",
                json={"query": "classified content"},
                headers=_auth_header(regular_user),
            )
            assert response.status_code == 200
            assert response.json()["citations"] == []
    finally:
        app.dependency_overrides.clear()
