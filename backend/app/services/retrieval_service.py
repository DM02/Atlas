import uuid
from dataclasses import dataclass

from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.reranker.base import RerankCandidate, RerankerProvider
from app.models.document import STATUS_READY, Document, DocumentChunk, DocumentVersion
from app.models.user import User
from app.repositories.document_repository import accessible_documents_clause


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    version_number: int
    page_number: int | None
    section_title: str | None
    content: str
    score: float


def _to_retrieved_chunk(
    row: Row[tuple[DocumentChunk, DocumentVersion, Document, float]],
) -> RetrievedChunk:
    chunk, version, document, score = row
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=document.id,
        document_title=document.title,
        version_number=version.version_number,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        content=chunk.content,
        score=score,
    )


async def _vector_candidates(
    session: AsyncSession, query_embedding: list[float], limit: int, user: User
) -> list[RetrievedChunk]:
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(DocumentChunk, DocumentVersion, Document, (1 - distance).label("score"))
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(
            Document.status == STATUS_READY,
            DocumentChunk.document_version_id == Document.current_version_id,
            accessible_documents_clause(user),
        )
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [_to_retrieved_chunk(row) for row in rows]


async def _fts_candidates(
    session: AsyncSession, query_text: str, limit: int, user: User
) -> list[RetrievedChunk]:
    """Postgres full-text search over the generated `content_tsv` column (GIN-indexed).

    English-only stemming/stopwording (`to_tsvector('english', ...)` at ingestion
    time) — a known limitation for multilingual corpora, not addressed in Phase 2.
    """
    tsquery = func.plainto_tsquery("english", query_text)
    rank = func.ts_rank_cd(DocumentChunk.content_tsv, tsquery)
    stmt = (
        select(DocumentChunk, DocumentVersion, Document, rank.label("score"))
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(
            Document.status == STATUS_READY,
            DocumentChunk.document_version_id == Document.current_version_id,
            DocumentChunk.content_tsv.op("@@")(tsquery),
            accessible_documents_clause(user),
        )
        .order_by(rank.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [_to_retrieved_chunk(row) for row in rows]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Combine independently-ranked chunk lists into one ranking (RRF).

    fused_score(chunk) = sum over lists containing it of 1 / (k + rank), where rank
    is 1-based position within that list. A chunk absent from a list contributes
    nothing from it — no penalty beyond simply not accumulating that term. k=60 is
    the standard constant from the original Cormack et al. RRF paper; it damps the
    influence of any single list's exact rank ordering without needing score
    normalization across methods that aren't on the same scale (cosine similarity
    vs. ts_rank_cd).
    """
    fused_scores: dict[uuid.UUID, float] = {}
    chunk_by_id: dict[uuid.UUID, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    fused: list[RetrievedChunk] = []
    for chunk_id in ordered_ids:
        chunk = chunk_by_id[chunk_id]
        chunk.score = fused_scores[chunk_id]
        fused.append(chunk)
    return fused


async def _apply_reranking(
    reranker: RerankerProvider, query: str, candidates: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    rerank_candidates = [RerankCandidate(id=str(c.chunk_id), text=c.content) for c in candidates]
    ranked = await reranker.rerank(query, rerank_candidates, top_n=top_n)

    by_id = {str(c.chunk_id): c for c in candidates}
    results = []
    for candidate_id, score in ranked:
        chunk = by_id[candidate_id]
        chunk.score = score
        results.append(chunk)
    return results


async def retrieve_relevant_chunks(
    *,
    session: AsyncSession,
    embedding_provider: EmbeddingProvider,
    query: str,
    top_k: int,
    requesting_user: User,
    use_hybrid_search: bool = False,
    use_reranking: bool = False,
    reranker: RerankerProvider | None = None,
    candidate_pool_size: int = 20,
) -> list[RetrievedChunk]:
    """Phase 1 baseline: plain pgvector cosine search, top_k only.
    Phase 2 additions (both opt-in via config, see docs/ARCHITECTURE.md §6, §9):
    - use_hybrid_search: fuse vector search with Postgres full-text search via RRF
    - use_reranking: rerank the candidate pool with a cross-encoder before truncating to top_k

    Phase 3: every candidate query is filtered by accessible_documents_clause, so a
    document requesting_user can't see never becomes a retrieval candidate in the
    first place — enforced in the SQL, not as a filter on already-fetched results
    (docs/ARCHITECTURE.md §10 risk 10.4).

    Phase 4: candidate queries only search chunks belonging to Document.current_version_id
    — a document with multiple versions never surfaces content from an old or
    still-processing version, only whatever's currently live.
    """
    query_embedding = await embedding_provider.embed_query(query)
    pool_size = candidate_pool_size if (use_hybrid_search or use_reranking) else top_k

    if use_hybrid_search:
        vector_candidates = await _vector_candidates(
            session, query_embedding, pool_size, requesting_user
        )
        fts_candidates = await _fts_candidates(session, query, pool_size, requesting_user)
        candidates = reciprocal_rank_fusion([vector_candidates, fts_candidates])
    else:
        candidates = await _vector_candidates(
            session, query_embedding, pool_size, requesting_user
        )

    if use_reranking and reranker is not None and candidates:
        return await _apply_reranking(reranker, query, candidates, top_n=top_k)

    return candidates[:top_k]
