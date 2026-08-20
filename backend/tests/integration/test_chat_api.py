from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient

from app.ai.embeddings.factory import get_embedding_provider
from app.ai.errors import AIProviderError
from app.ai.llm.factory import get_llm_provider
from app.ai.reranker.base import RerankCandidate
from app.ai.reranker.factory import get_reranker
from app.core.config import Settings, get_settings
from app.core.queue import get_arq_pool
from app.core.security import create_access_token
from app.core.storage import LocalStorageBackend, get_storage_backend
from app.db.session import get_db
from app.main import app
from tests.integration.factories import ImmediateArqPool


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
        return "Atlas is a RAG platform [1]."


class FailingEmbeddingProvider:
    model_name = "failing-embed"
    dimension = 8

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AIProviderError("simulated provider outage")

    async def embed_query(self, text: str) -> list[float]:
        raise AIProviderError("simulated provider outage")


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


async def test_chat_query_without_documents_refuses(db_session, tmp_path, admin_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/query",
                json={"query": "What is Atlas?"},
                headers=_auth_header(admin_user),
            )

            assert response.status_code == 200
            body = response.json()
            assert "don't have enough information" in body["answer"]
            assert body["citations"] == []
    finally:
        app.dependency_overrides.clear()


async def test_chat_query_with_document_returns_grounded_citation(
    db_session, tmp_path, admin_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"Atlas is a RAG platform. " * 20, "text/plain")},
                headers=headers,
            )

            response = await client.post(
                "/api/v1/chat/query", json={"query": "What is Atlas?"}, headers=headers
            )
            assert response.status_code == 200
            body = response.json()
            assert body["answer"] == "Atlas is a RAG platform [1]."
            assert len(body["citations"]) == 1
            assert body["citations"][0]["document_title"] == "notes.txt"

            conversation_id = body["conversation_id"]

            listing = await client.get("/api/v1/chat/conversations", headers=headers)
            assert listing.status_code == 200
            assert any(c["id"] == conversation_id for c in listing.json())

            detail = await client.get(
                f"/api/v1/chat/conversations/{conversation_id}", headers=headers
            )
            assert detail.status_code == 200
            messages = detail.json()["messages"]
            assert [m["role"] for m in messages] == ["user", "assistant"]
            assert len(messages[1]["citations"]) == 1

            follow_up = await client.post(
                "/api/v1/chat/query",
                json={"query": "Tell me more", "conversation_id": conversation_id},
                headers=headers,
            )
            assert follow_up.status_code == 200
            assert follow_up.json()["conversation_id"] == conversation_id
    finally:
        app.dependency_overrides.clear()


async def test_chat_query_with_unknown_conversation_id_returns_404(
    db_session, tmp_path, admin_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/query",
                json={
                    "query": "hi",
                    "conversation_id": "00000000-0000-0000-0000-000000000000",
                },
                headers=_auth_header(admin_user),
            )
            assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


class NoOpReranker:
    """Keeps candidate order as-is — just proves get_reranker() gets wired in and
    called when enable_reranking=True, without depending on the real model.
    """

    model_name = "noop-fake"

    async def rerank(
        self, query: str, candidates: list[RerankCandidate], top_n: int
    ) -> list[tuple[str, float]]:
        return [(c.id, 1.0) for c in candidates[:top_n]]


async def test_chat_query_with_hybrid_search_and_reranking_enabled(
    db_session, tmp_path, admin_user
) -> None:
    settings = _wire_overrides(db_session, tmp_path)
    settings.enable_hybrid_search = True
    settings.enable_reranking = True
    app.dependency_overrides[get_reranker] = lambda: NoOpReranker()
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/documents",
                files={"file": ("notes.txt", b"Atlas is a RAG platform. " * 20, "text/plain")},
                headers=headers,
            )

            response = await client.post(
                "/api/v1/chat/query", json={"query": "What is Atlas?"}, headers=headers
            )
            assert response.status_code == 200
            body = response.json()
            assert body["answer"] == "Atlas is a RAG platform [1]."
            assert len(body["citations"]) == 1
    finally:
        app.dependency_overrides.clear()


async def test_chat_query_returns_503_when_ai_provider_fails(
    db_session, tmp_path, admin_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    app.dependency_overrides[get_embedding_provider] = lambda: FailingEmbeddingProvider()
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/query",
                json={"query": "What is Atlas?"},
                headers=_auth_header(admin_user),
            )
            assert response.status_code == 503
            assert "unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
