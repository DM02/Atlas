"""Phase 5/6 generation-quality experiment: runs the REAL RAG pipeline
(retrieval + LLM generation) over all 30 golden questions and computes
answer_correctness / citation_correctness / groundedness_rate /
correct_refusal_rate / hallucination_rate for real.

This was the one part of Phase 5's evaluation blocked by OpenAI billing (see
docs/EVALUATION.md's Limitations) — unblocked in Phase 6 by routing the LLM
(and embeddings) through OpenRouter, which has free-tier chat models even
though this project's OpenAI account doesn't (see core/config.py's
llm_base_url / embedding_base_url).

Uses the REAL production embedding_provider and llm provider (not eval's
ZeroPaddedEmbeddingProvider workaround from pgvector_experiments.py) — this
run reflects the actual production pipeline end to end, not an approximation.

Usage:
    backend/.venv/Scripts/python -m eval.runners.generation_experiment
(needs OPENROUTER_API_KEY set; takes several minutes on a free-tier model)
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

from app.ai.embeddings.factory import get_embedding_provider
from app.ai.llm.factory import get_llm_provider
from app.ai.pipeline.rag_pipeline import NO_ANSWER_TEXT, generate_answer
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
from eval.metrics.generation import (
    answer_correctness,
    citation_correctness,
    correct_refusal_rate,
    groundedness_rate,
    hallucination_rate,
    has_citation_when_answering,
)
from eval.runners.common import CORPUS_DIR, GoldenQuestion, load_golden_questions, write_report

EVAL_USER_EMAIL = "eval-runner@atlas.internal"


@dataclass
class GenerationSummary:
    run_id: uuid.UUID
    num_questions: int
    answer_correctness_rate: float
    citation_correctness_rate: float
    groundedness_rate: float
    correct_refusal_rate: float
    hallucination_rate: float
    mean_latency_ms: float


async def _ensure_eval_user(session: AsyncSession) -> User:
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
    provider = get_embedding_provider()

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


async def run_generation_experiment(
    session: AsyncSession, questions: list[GoldenQuestion], user: User
) -> GenerationSummary:
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    llm = get_llm_provider()

    run = EvaluationRun(
        name="generation_quality",
        config={
            "experiment": "generation_quality",
            "llm_model": llm.model_name,
            "embedding_model": embedding_provider.model_name,
        },
        status=STATUS_RUNNING,
    )
    session.add(run)
    await session.flush()

    answer_correct_flags: list[bool] = []
    citation_correct_flags: list[bool] = []
    answerable_pairs: list[tuple[str, list[int]]] = []
    unanswerable_answers: list[str] = []
    latencies_ms: list[float] = []

    for question in questions:
        start = time.perf_counter()
        chunks = await retrieve_relevant_chunks(
            session=session,
            embedding_provider=embedding_provider,
            query=question.question,
            top_k=settings.retrieval_top_k,
            requesting_user=user,
            use_hybrid_search=settings.enable_hybrid_search,
            use_reranking=settings.enable_reranking,
            candidate_pool_size=settings.retrieval_candidate_pool_size,
        )
        generation = await generate_answer(llm=llm, query=question.question, chunks=chunks)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        cited_documents = {chunks[i - 1].document_title for i in generation.cited_chunk_indices}

        if question.answerable:
            correct = answer_correctness(generation.answer, question.expected_answer_contains)
            cite_correct = citation_correctness(cited_documents, question.expected_source_documents)
            grounded = has_citation_when_answering(
                generation.answer, generation.cited_chunk_indices, NO_ANSWER_TEXT
            )
            answer_correct_flags.append(correct)
            citation_correct_flags.append(cite_correct)
            answerable_pairs.append((generation.answer, generation.cited_chunk_indices))
            metrics = {
                "answer_correct": correct,
                "citation_correct": cite_correct,
                "grounded": grounded,
            }
        else:
            unanswerable_answers.append(generation.answer)
            metrics = {"refused": generation.answer.strip() == NO_ANSWER_TEXT.strip()}

        session.add(
            EvaluationResult(
                evaluation_run_id=run.id,
                question_id=question.id,
                retrieved_chunk_ids=[str(c.chunk_id) for c in chunks],
                metrics=metrics,
                latency_ms=int(elapsed_ms),
            )
        )
        await session.commit()
        print(f"  {question.id}: done ({elapsed_ms:.0f} ms)")

    run.status = STATUS_COMPLETED
    run.finished_at = datetime.now(UTC)
    await session.commit()

    return GenerationSummary(
        run_id=run.id,
        num_questions=len(questions),
        answer_correctness_rate=mean(answer_correct_flags) if answer_correct_flags else 0.0,
        citation_correctness_rate=mean(citation_correct_flags) if citation_correct_flags else 0.0,
        groundedness_rate=groundedness_rate(answerable_pairs, NO_ANSWER_TEXT),
        correct_refusal_rate=correct_refusal_rate(unanswerable_answers, NO_ANSWER_TEXT),
        hallucination_rate=hallucination_rate(unanswerable_answers, NO_ANSWER_TEXT),
        mean_latency_ms=mean(latencies_ms) if latencies_ms else 0.0,
    )


async def main() -> None:
    questions = load_golden_questions()

    async with async_session_factory() as session:
        user = await _ensure_eval_user(session)
        await _ensure_corpus_ingested(session, user)
        print(f"Running generation experiment over {len(questions)} questions...")
        summary = await run_generation_experiment(session, questions, user)

    print(f"\nGeneration-quality experiment ({summary.num_questions} questions)")
    print(f"  answer_correctness_rate:  {summary.answer_correctness_rate:.3f}")
    print(f"  citation_correctness_rate:{summary.citation_correctness_rate:.3f}")
    print(f"  groundedness_rate:        {summary.groundedness_rate:.3f}")
    print(f"  correct_refusal_rate:     {summary.correct_refusal_rate:.3f}")
    print(f"  hallucination_rate:       {summary.hallucination_rate:.3f}")
    print(f"  mean latency:             {summary.mean_latency_ms:.1f} ms")

    report_path = write_report("generation_experiment", summary.__dict__)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
