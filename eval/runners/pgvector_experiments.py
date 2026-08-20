"""Phase 5 experiment runner: hybrid search vs vector-only, and reranking
on vs off — both run through the REAL pgvector-backed retrieval_service
(unlike eval/runners/in_memory_experiments.py's embedding/chunking
comparison), since neither dimension varies embedding output size, so the
production code path applies unmodified.

The golden corpus is ingested through the real ingestion_service
(create_pending_version + process_document_version) into a real Postgres,
using a local sentence-transformers model wrapped in
eval.runners.common.ZeroPaddedEmbeddingProvider to satisfy
DocumentChunk.embedding's fixed Vector(1536) column without needing live
OpenAI access — see that class's docstring for why the padding doesn't
distort retrieval ranking. Reranking uses the real local cross-encoder
(app.ai.reranker.cross_encoder.CrossEncoderReranker) — no OpenAI needed there
either.

Usage:
    backend/.venv/Scripts/python -m eval.runners.pgvector_experiments
(run from the repo root; needs a reachable Postgres per DATABASE_URL — the
same one docker-compose's `postgres` service points at)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.ai.reranker.factory import get_reranker
from app.core.config import get_settings
from app.core.security import hash_password
from app.core.storage import get_storage_backend
from app.db.session import async_session_factory
from app.models.document import Document
from app.models.evaluation import STATUS_COMPLETED, STATUS_RUNNING, EvaluationResult, EvaluationRun
from app.models.user import ROLE_ADMIN, User
from app.repositories.user_repository import get_role_by_name, get_user_by_email
from app.services.ingestion_service import create_pending_version, process_document_version
from app.services.retrieval_service import retrieve_relevant_chunks
from eval.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from eval.runners.common import (
    CORPUS_DIR,
    GoldenQuestion,
    ZeroPaddedEmbeddingProvider,
    dedupe_preserve_order,
    load_golden_questions,
    write_report,
)

EVAL_USER_EMAIL = "eval-runner@atlas.internal"
# Local model deliberately fixed (not compared here) — this experiment isolates
# retrieval-time behavior (hybrid/rerank), not embedding choice; see
# in_memory_experiments.py for the embedding-model comparison.
EVAL_EMBEDDING_MODEL = ("BAAI/bge-small-en-v1.5", 384)

RETRIEVAL_VARIANTS: list[tuple[str, bool, bool]] = [
    # (name, use_hybrid_search, use_reranking)
    ("vector_only", False, False),
    ("hybrid_rrf", True, False),
    ("vector_reranked", False, True),
    ("hybrid_reranked", True, True),
]


@dataclass
class ExperimentSummary:
    run_id: uuid.UUID
    run_name: str
    mean_recall_at_k: float
    mean_precision_at_k: float
    mrr: float
    mean_latency_ms: float
    num_questions: int


async def _ensure_eval_user(session: AsyncSession) -> User:
    # accessible_documents_clause() reads user.role.name, so the role
    # relationship must be eager-loaded — get_user_by_email() already does
    # this (selectinload); a plain select(User) here would lazy-load it and
    # crash under SQLAlchemy's async engine (MissingGreenlet).
    existing = await get_user_by_email(session, EVAL_USER_EMAIL)
    if existing is not None:
        return existing

    admin_role = await get_role_by_name(session, ROLE_ADMIN)
    if admin_role is None:
        raise RuntimeError("Admin role not seeded — run `alembic upgrade head` first")

    user = User(
        email=EVAL_USER_EMAIL,
        hashed_password=hash_password(uuid.uuid4().hex),
        role_id=admin_role.id,
    )
    session.add(user)
    await session.commit()
    return await get_user_by_email(session, EVAL_USER_EMAIL)


async def _ensure_corpus_ingested(session: AsyncSession, owner: User) -> None:
    settings = get_settings()
    storage = get_storage_backend()
    provider = ZeroPaddedEmbeddingProvider(
        SentenceTransformerEmbeddingProvider(*EVAL_EMBEDDING_MODEL),
        target_dimension=settings.openai_embedding_dimension,
    )

    for path in sorted(CORPUS_DIR.glob("*.txt")):
        existing = (
            await session.execute(
                select(Document).where(Document.owner_id == owner.id, Document.title == path.name)
            )
        ).scalar_one_or_none()
        if existing is not None and existing.current_version_id is not None:
            continue

        version = await create_pending_version(
            session=session,
            storage=storage,
            settings=settings,
            filename=path.name,
            content=path.read_bytes(),
            owner_id=owner.id,
            document=existing,
        )
        await process_document_version(
            session=session,
            storage=storage,
            embedding_provider=provider,
            settings=settings,
            document_version_id=version.id,
        )


async def _run_retrieval_variant(
    *,
    session: AsyncSession,
    run_name: str,
    config: dict,
    query_provider: ZeroPaddedEmbeddingProvider,
    questions: list[GoldenQuestion],
    user: User,
    use_hybrid_search: bool,
    use_reranking: bool,
    top_k: int,
    candidate_pool_size: int,
) -> ExperimentSummary:
    answerable = [q for q in questions if q.answerable]
    reranker = get_reranker() if use_reranking else None

    run = EvaluationRun(name=run_name, config=config, status=STATUS_RUNNING)
    session.add(run)
    await session.flush()

    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for question in answerable:
        start = time.perf_counter()
        chunks = await retrieve_relevant_chunks(
            session=session,
            embedding_provider=query_provider,
            query=question.question,
            top_k=top_k,
            requesting_user=user,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            reranker=reranker,
            candidate_pool_size=candidate_pool_size,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        retrieved_documents = dedupe_preserve_order(c.document_title for c in chunks)
        relevant = set(question.expected_source_documents)

        recall = recall_at_k(retrieved_documents, relevant, k=top_k)
        precision = precision_at_k(retrieved_documents, relevant, k=top_k)
        rr = reciprocal_rank(retrieved_documents, relevant)

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(rr)
        latencies_ms.append(elapsed_ms)

        session.add(
            EvaluationResult(
                evaluation_run_id=run.id,
                question_id=question.id,
                retrieved_chunk_ids=[str(c.chunk_id) for c in chunks],
                metrics={"recall_at_k": recall, "precision_at_k": precision, "reciprocal_rank": rr},
                latency_ms=int(elapsed_ms),
            )
        )

    run.status = STATUS_COMPLETED
    run.finished_at = datetime.now(UTC)
    await session.commit()

    return ExperimentSummary(
        run_id=run.id,
        run_name=run_name,
        mean_recall_at_k=mean(recalls) if recalls else 0.0,
        mean_precision_at_k=mean(precisions) if precisions else 0.0,
        mrr=mean_reciprocal_rank(reciprocal_ranks),
        mean_latency_ms=mean(latencies_ms) if latencies_ms else 0.0,
        num_questions=len(answerable),
    )


async def run_hybrid_and_reranking_comparison(
    session: AsyncSession, questions: list[GoldenQuestion], user: User
) -> list[ExperimentSummary]:
    settings = get_settings()
    query_provider = ZeroPaddedEmbeddingProvider(
        SentenceTransformerEmbeddingProvider(*EVAL_EMBEDDING_MODEL),
        target_dimension=settings.openai_embedding_dimension,
    )

    summaries = []
    for name, use_hybrid_search, use_reranking in RETRIEVAL_VARIANTS:
        summary = await _run_retrieval_variant(
            session=session,
            run_name=f"hybrid_rerank_comparison:{name}",
            config={
                "experiment": "hybrid_rerank_comparison",
                "variant": name,
                "use_hybrid_search": use_hybrid_search,
                "use_reranking": use_reranking,
                "embedding_model": EVAL_EMBEDDING_MODEL[0],
            },
            query_provider=query_provider,
            questions=questions,
            user=user,
            use_hybrid_search=use_hybrid_search,
            use_reranking=use_reranking,
            top_k=settings.retrieval_top_k,
            candidate_pool_size=settings.retrieval_candidate_pool_size,
        )
        summaries.append(summary)
    return summaries


def _print_summary(title: str, summaries: list[ExperimentSummary]) -> None:
    print(f"\n{title}")
    print(f"{'run':45s} {'recall@k':>10s} {'precision@k':>12s} {'mrr':>8s} {'avg ms':>8s}")
    for s in summaries:
        print(
            f"{s.run_name:45s} {s.mean_recall_at_k:>10.3f} {s.mean_precision_at_k:>12.3f} "
            f"{s.mrr:>8.3f} {s.mean_latency_ms:>8.1f}"
        )


async def main() -> None:
    questions = load_golden_questions()

    async with async_session_factory() as session:
        user = await _ensure_eval_user(session)
        await _ensure_corpus_ingested(session, user)
        summaries = await run_hybrid_and_reranking_comparison(session, questions, user)

    _print_summary("Hybrid search / reranking comparison", summaries)

    report_path = write_report(
        "pgvector_experiments", {"hybrid_rerank_comparison": [s.__dict__ for s in summaries]}
    )
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
