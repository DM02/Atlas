"""Phase 5 experiment runner: embedding-model comparison and chunking-strategy
comparison, both computed with in-memory numpy cosine similarity rather than
through the real pgvector-backed retrieval_service.

Why in-memory: DocumentChunk.embedding is a fixed 1536-dim pgvector column
(matching OpenAI's text-embedding-3-small), so two local sentence-transformers
models with different output dimensions (384 vs 768) can't both be written to
that column without a schema migration per model — see
docs/ARCHITECTURE.md §10 risk 10.2. Comparing them in-memory sidesteps that
limitation entirely and needs no OpenAI credits (the project's OpenAI account
is billing-blocked).

Ground truth in golden_qa.yaml is at DOCUMENT granularity, so both
experiments dedupe ranked chunks down to their source documents before
scoring recall/precision/MRR — see eval/metrics/retrieval.py's module
docstring for why.

Usage:
    backend/.venv/Scripts/python -m eval.runners.in_memory_experiments
(run from the repo root so the `app` and `eval` packages both resolve; needs
a reachable Postgres per DATABASE_URL to persist EvaluationRun/EvaluationResult rows)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.ai.pipeline.chunking import Chunk, chunk_by_headings, chunk_pages
from app.ai.pipeline.extraction import ExtractedPage
from app.core.config import get_settings
from app.db.session import async_session_factory
from app.models.evaluation import STATUS_COMPLETED, STATUS_RUNNING, EvaluationResult, EvaluationRun
from eval.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from eval.runners.common import (
    CorpusDocument,
    GoldenQuestion,
    dedupe_preserve_order,
    embed_all,
    load_corpus,
    load_golden_questions,
    rank_by_cosine_similarity,
    write_report,
)

# BAAI/bge-small-en-v1.5 (384-dim) is the fixed embedding model for the
# chunking-strategy comparison — only the chunker varies there, so holding
# the embedding model constant keeps that comparison isolated to one variable.
EMBEDDING_MODELS: list[tuple[str, int]] = [
    ("BAAI/bge-small-en-v1.5", 384),
    ("BAAI/bge-base-en-v1.5", 768),
]
CHUNKING_STRATEGIES: list[tuple[str, Callable[[list[ExtractedPage], int, int], list[Chunk]]]] = [
    ("fixed_size", chunk_pages),
    ("structure_aware", chunk_by_headings),
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


def _build_chunks(
    corpus: list[CorpusDocument],
    chunker: Callable[[list[ExtractedPage], int, int], list[Chunk]],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, Chunk]]:
    """Chunks each document independently (matching production, which chunks
    one document's pages at a time) and tags each Chunk with its source
    document filename."""
    result: list[tuple[str, Chunk]] = []
    for document in corpus:
        for chunk in chunker(document.pages, max_tokens, overlap_tokens):
            result.append((document.filename, chunk))
    return result


async def _run_retrieval_experiment(
    *,
    session: AsyncSession,
    run_name: str,
    config: dict,
    chunks: list[tuple[str, Chunk]],
    provider: EmbeddingProvider,
    questions: list[GoldenQuestion],
    top_k: int,
) -> ExperimentSummary:
    answerable = [q for q in questions if q.answerable]
    chunk_vectors = await embed_all(provider, [chunk.content for _, chunk in chunks])

    run = EvaluationRun(name=run_name, config=config, status=STATUS_RUNNING)
    session.add(run)
    await session.flush()

    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []

    for question in answerable:
        start = time.perf_counter()
        query_vector = (await embed_all(provider, [question.question]))[0]
        ranked_indices = rank_by_cosine_similarity(query_vector, chunk_vectors)
        top_indices = ranked_indices[:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        retrieved_documents = dedupe_preserve_order(chunks[i][0] for i in top_indices)
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
                retrieved_chunk_ids=[
                    f"{chunks[i][0]}#{chunks[i][1].chunk_index}" for i in top_indices
                ],
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


async def run_embedding_comparison(
    session: AsyncSession, corpus: list[CorpusDocument], questions: list[GoldenQuestion]
) -> list[ExperimentSummary]:
    settings = get_settings()
    chunks = _build_chunks(
        corpus, chunk_pages, settings.chunk_max_tokens, settings.chunk_overlap_tokens
    )

    summaries = []
    for model_name, dimension in EMBEDDING_MODELS:
        provider = SentenceTransformerEmbeddingProvider(model_name, dimension)
        summary = await _run_retrieval_experiment(
            session=session,
            run_name=f"embedding_comparison:{model_name}",
            config={
                "experiment": "embedding_comparison",
                "embedding_model": model_name,
                "dimension": dimension,
                "chunking_strategy": "fixed_size",
            },
            chunks=chunks,
            provider=provider,
            questions=questions,
            top_k=settings.retrieval_top_k,
        )
        summaries.append(summary)
    return summaries


async def run_chunking_comparison(
    session: AsyncSession, corpus: list[CorpusDocument], questions: list[GoldenQuestion]
) -> list[ExperimentSummary]:
    settings = get_settings()
    model_name, dimension = EMBEDDING_MODELS[0]
    provider = SentenceTransformerEmbeddingProvider(model_name, dimension)

    summaries = []
    for strategy_name, chunker in CHUNKING_STRATEGIES:
        chunks = _build_chunks(
            corpus, chunker, settings.chunk_max_tokens, settings.chunk_overlap_tokens
        )
        summary = await _run_retrieval_experiment(
            session=session,
            run_name=f"chunking_comparison:{strategy_name}",
            config={
                "experiment": "chunking_comparison",
                "chunking_strategy": strategy_name,
                "embedding_model": model_name,
            },
            chunks=chunks,
            provider=provider,
            questions=questions,
            top_k=settings.retrieval_top_k,
        )
        summaries.append(summary)
    return summaries


def _print_summary(title: str, summaries: list[ExperimentSummary]) -> None:
    print(f"\n{title}")
    print(f"{'run':40s} {'recall@k':>10s} {'precision@k':>12s} {'mrr':>8s} {'avg ms':>8s}")
    for s in summaries:
        print(
            f"{s.run_name:40s} {s.mean_recall_at_k:>10.3f} {s.mean_precision_at_k:>12.3f} "
            f"{s.mrr:>8.3f} {s.mean_latency_ms:>8.1f}"
        )


async def main() -> None:
    corpus = load_corpus()
    questions = load_golden_questions()

    async with async_session_factory() as session:
        embedding_summaries = await run_embedding_comparison(session, corpus, questions)
        chunking_summaries = await run_chunking_comparison(session, corpus, questions)

    _print_summary("Embedding model comparison", embedding_summaries)
    _print_summary("Chunking strategy comparison", chunking_summaries)

    report_path = write_report(
        "in_memory_experiments",
        {
            "embedding_comparison": [s.__dict__ for s in embedding_summaries],
            "chunking_comparison": [s.__dict__ for s in chunking_summaries],
        },
    )
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
