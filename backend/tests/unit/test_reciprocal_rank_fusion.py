import uuid

from app.services.retrieval_service import RetrievedChunk, reciprocal_rank_fusion


def _chunk(score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="doc",
        version_number=1,
        page_number=None,
        section_title=None,
        content="content",
        score=score,
    )


def test_rrf_ranks_chunk_present_in_both_lists_above_single_list_chunk() -> None:
    shared = _chunk()
    vector_only = _chunk()
    fts_only = _chunk()

    # shared is rank 2 in both lists; vector_only is rank 1 in vector list only.
    vector_list = [vector_only, shared]
    fts_list = [fts_only, shared]

    fused = reciprocal_rank_fusion([vector_list, fts_list])

    assert fused[0].chunk_id == shared.chunk_id


def test_rrf_deduplicates_chunk_present_in_multiple_lists() -> None:
    shared = _chunk()
    fused = reciprocal_rank_fusion([[shared], [shared]])

    assert len(fused) == 1
    assert fused[0].chunk_id == shared.chunk_id


def test_rrf_preserves_order_for_a_single_list() -> None:
    first, second, third = _chunk(), _chunk(), _chunk()
    fused = reciprocal_rank_fusion([[first, second, third]])

    assert [c.chunk_id for c in fused] == [first.chunk_id, second.chunk_id, third.chunk_id]


def test_rrf_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_score_matches_formula() -> None:
    chunk = _chunk()
    fused = reciprocal_rank_fusion([[chunk]], k=60)

    assert fused[0].score == 1 / (60 + 1)
