from sqlalchemy import select

from app.core.config import Settings
from app.core.storage import LocalStorageBackend
from app.models.metrics import RequestMetric
from tests.integration.factories import ingest_test_document


class FakeEmbeddingProvider:
    model_name = "fake-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


async def test_successful_ingestion_persists_request_metric_with_stage_breakdown(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="notes.txt",
        content=b"Atlas is a RAG platform. " * 50,
        owner_id=admin_user.id,
    )

    metrics = (
        (
            await db_session.execute(
                select(RequestMetric).where(RequestMetric.endpoint == "document_ingestion")
            )
        )
        .scalars()
        .all()
    )

    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.total_ms >= 0
    assert set(metric.stage_latencies_ms.keys()) == {"extract_ms", "chunk_ms", "embed_ms"}
    assert all(v >= 0 for v in metric.stage_latencies_ms.values())


async def test_failed_ingestion_does_not_persist_request_metric(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = FakeEmbeddingProvider(settings.openai_embedding_dimension)

    # A .pdf extension with content pypdf can't parse fails extraction —
    # the pipeline should record nothing for the (never-completed) stages.
    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="broken.pdf",
        content=b"this is not a real pdf file",
        owner_id=admin_user.id,
    )

    metrics = (
        (
            await db_session.execute(
                select(RequestMetric).where(RequestMetric.endpoint == "document_ingestion")
            )
        )
        .scalars()
        .all()
    )
    assert metrics == []
