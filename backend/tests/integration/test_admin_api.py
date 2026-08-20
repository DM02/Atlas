from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from httpx import ASGITransport, AsyncClient

from app.ai.embeddings.factory import get_embedding_provider
from app.ai.llm.factory import get_llm_provider
from app.core.config import Settings, get_settings
from app.core.queue import get_arq_pool
from app.core.security import create_access_token
from app.core.storage import LocalStorageBackend, get_storage_backend
from app.db.session import get_db
from app.main import app
from app.models.evaluation import STATUS_COMPLETED, EvaluationResult, EvaluationRun
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


async def test_admin_metrics_requires_authentication(db_session, tmp_path) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/admin/metrics")
            assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_admin_metrics_rejects_non_admin_user(db_session, tmp_path, regular_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/admin/metrics", headers=_auth_header(regular_user)
            )
            assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()


async def test_admin_metrics_reflects_real_chat_query_latency(
    db_session, tmp_path, admin_user
) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)
    headers = _auth_header(admin_user)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Two real chat queries through the full pipeline — chat_service
            # persists a RequestMetric row for each (see services/chat_service.py).
            for _ in range(2):
                query_response = await client.post(
                    "/api/v1/chat/query", json={"query": "What is Atlas?"}, headers=headers
                )
                assert query_response.status_code == 200

            response = await client.get("/api/v1/admin/metrics", headers=headers)
            assert response.status_code == 200
            body = response.json()

            chat_stats = next(e for e in body["endpoints"] if e["endpoint"] == "chat_query")
            assert chat_stats["count"] == 2
            assert chat_stats["p50_total_ms"] >= 0
            assert chat_stats["p95_total_ms"] >= chat_stats["p50_total_ms"]

            assert len(body["recent"]) == 2
            assert "retrieve_ms" in body["recent"][0]["stage_latencies_ms"]
            assert "generate_ms" in body["recent"][0]["stage_latencies_ms"]
    finally:
        app.dependency_overrides.clear()


async def test_admin_evaluations_rejects_non_admin_user(db_session, tmp_path, regular_user) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/admin/evaluations", headers=_auth_header(regular_user)
            )
            assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()


async def test_admin_evaluations_lists_runs_with_averaged_metrics(
    db_session, tmp_path, admin_user
) -> None:
    _wire_overrides(db_session, tmp_path)

    run = EvaluationRun(
        name="test_run",
        config={"experiment": "unit_test"},
        status=STATUS_COMPLETED,
        finished_at=datetime.now(UTC),
    )
    db_session.add(run)
    await db_session.flush()
    db_session.add_all(
        [
            EvaluationResult(
                evaluation_run_id=run.id,
                question_id="q1",
                retrieved_chunk_ids=[],
                metrics={"recall_at_k": 1.0, "precision_at_k": 0.5},
                latency_ms=10,
            ),
            EvaluationResult(
                evaluation_run_id=run.id,
                question_id="q2",
                retrieved_chunk_ids=[],
                metrics={"recall_at_k": 0.0, "precision_at_k": 0.25},
                latency_ms=20,
            ),
        ]
    )
    await db_session.commit()

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/admin/evaluations", headers=_auth_header(admin_user)
            )
            assert response.status_code == 200
            body = response.json()

            run_out = next(r for r in body if r["name"] == "test_run")
            assert run_out["result_count"] == 2
            assert run_out["mean_metrics"]["recall_at_k"] == 0.5
            assert run_out["mean_metrics"]["precision_at_k"] == 0.375
    finally:
        app.dependency_overrides.clear()
