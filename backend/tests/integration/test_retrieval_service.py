from app.ai.reranker.base import RerankCandidate
from app.ai.reranker.cross_encoder import CrossEncoderReranker
from app.core.config import Settings
from app.core.storage import LocalStorageBackend
from app.services.retrieval_service import retrieve_relevant_chunks
from tests.integration.factories import ingest_test_document


class KeywordEmbeddingProvider:
    """Deterministic fake: dim 0 lights up for 'apple', dim 1 for 'orange'.

    Lets a test assert real cosine-distance ordering from Postgres/pgvector
    without depending on the real OpenAI embedding API.
    """

    model_name = "fake-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def _vector_for(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        lowered = text.lower()
        if "apple" in lowered:
            vec[0] = 1.0
        if "orange" in lowered:
            vec[1] = 1.0
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)


async def test_retrieve_relevant_chunks_orders_by_cosine_similarity(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    for filename, word in [("apple.txt", "apple"), ("orange.txt", "orange")]:
        document = await ingest_test_document(
            session=db_session,
            storage=storage,
            embedding_provider=embeddings,
            settings=settings,
            filename=filename,
            content=(word + " ").encode() * 20,
            owner_id=admin_user.id,
        )
        assert document.status == "ready"

    results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=5,
        requesting_user=admin_user,
    )

    assert len(results) == 2
    assert results[0].document_title == "apple.txt"
    assert results[0].score > results[1].score


async def test_retrieve_relevant_chunks_respects_top_k(db_session, tmp_path, admin_user) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    for filename, word in [("apple.txt", "apple"), ("orange.txt", "orange")]:
        await ingest_test_document(
            session=db_session,
            storage=storage,
            embedding_provider=embeddings,
            settings=settings,
            filename=filename,
            content=(word + " ").encode() * 20,
            owner_id=admin_user.id,
        )

    results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=1,
        requesting_user=admin_user,
    )

    assert len(results) == 1


class MisleadingEmbeddingProvider:
    """Deliberately embeds every query as the 'banana' vector, regardless of the
    query text — a stand-in for an embedding model that's confidently wrong.
    Used to prove hybrid search's full-text component can override a bad vector
    match, not just ride along with it.
    """

    model_name = "misleading-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimension
            lowered = text.lower()
            if "banana" in lowered:
                vec[0] = 1.0
            if "apple" in lowered:
                vec[1] = 1.0
            vectors.append(vec)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        vec[0] = 1.0  # always points at "banana", even when asked about apples
        return vec


async def test_hybrid_search_lets_fulltext_override_a_bad_vector_match(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = MisleadingEmbeddingProvider(settings.openai_embedding_dimension)

    for filename, word in [("banana.txt", "banana"), ("apple.txt", "apple")]:
        await ingest_test_document(
            session=db_session,
            storage=storage,
            embedding_provider=embeddings,
            settings=settings,
            filename=filename,
            content=(word + " ").encode() * 20,
            owner_id=admin_user.id,
        )

    vector_only = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=2,
        use_hybrid_search=False,
        requesting_user=admin_user,
    )
    assert vector_only[0].document_title == "banana.txt"  # the misleading embedding wins alone

    hybrid = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=2,
        use_hybrid_search=True,
        candidate_pool_size=10,
        requesting_user=admin_user,
    )
    assert hybrid[0].document_title == "apple.txt"  # full-text match rescues the right answer


class ReversingReranker:
    """Fake reranker that just reverses candidate order — enough to prove the
    retrieval pipeline actually applies whatever the reranker returns, without
    depending on the real cross-encoder's judgment.
    """

    model_name = "reversing-fake"

    async def rerank(
        self, query: str, candidates: list[RerankCandidate], top_n: int
    ) -> list[tuple[str, float]]:
        reversed_candidates = list(reversed(candidates))
        return [(c.id, float(i)) for i, c in enumerate(reversed_candidates)][:top_n]


async def test_reranking_applies_reranker_order_and_respects_top_k(
    db_session, tmp_path, admin_user
) -> None:
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = KeywordEmbeddingProvider(settings.openai_embedding_dimension)

    for filename, word in [("apple.txt", "apple"), ("orange.txt", "orange")]:
        await ingest_test_document(
            session=db_session,
            storage=storage,
            embedding_provider=embeddings,
            settings=settings,
            filename=filename,
            content=(word + " ").encode() * 20,
            owner_id=admin_user.id,
        )

    # Plain vector search would rank apple.txt first (query="apple").
    baseline = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=2,
        requesting_user=admin_user,
    )
    assert baseline[0].document_title == "apple.txt"

    reranked = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="apple",
        top_k=1,
        use_reranking=True,
        reranker=ReversingReranker(),
        candidate_pool_size=10,
        requesting_user=admin_user,
    )
    assert len(reranked) == 1
    assert reranked[0].document_title == "orange.txt"  # reranker's reversal flipped the order


class ConstantEmbeddingProvider:
    """Every text gets the same non-zero vector — vector search becomes a tie,
    so whichever chunk wins is entirely down to the reranker, not the embedder.
    (An all-keyword provider like KeywordEmbeddingProvider would give unrelated
    text a zero vector here, and cosine distance on a zero vector is undefined.)
    """

    model_name = "constant-embed"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def _vector(self) -> list[float]:
        vec = [0.0] * self.dimension
        vec[0] = 1.0
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector() for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector()


async def test_real_cross_encoder_reranks_relevant_chunk_first(
    db_session, tmp_path, admin_user
) -> None:
    """Exercises the actual BAAI/bge-reranker-base model (downloaded/cached on
    first run), not a fake — proves the abstraction genuinely works, not just
    the wiring around it. Slower than the rest of the suite by design.
    """
    settings = Settings(storage_path=str(tmp_path))
    storage = LocalStorageBackend(str(tmp_path))
    embeddings = ConstantEmbeddingProvider(settings.openai_embedding_dimension)

    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="atlas.txt",
        content=b"Atlas is a production-oriented RAG platform focused on retrieval quality.",
        owner_id=admin_user.id,
    )
    await ingest_test_document(
        session=db_session,
        storage=storage,
        embedding_provider=embeddings,
        settings=settings,
        filename="unrelated.txt",
        content=b"Bananas are a good source of potassium and dietary fiber.",
        owner_id=admin_user.id,
    )

    reranker = CrossEncoderReranker(model_name=settings.reranker_model)
    results = await retrieve_relevant_chunks(
        session=db_session,
        embedding_provider=embeddings,
        query="What is Atlas?",
        top_k=1,
        use_reranking=True,
        reranker=reranker,
        candidate_pool_size=10,
        requesting_user=admin_user,
    )

    assert results[0].document_title == "atlas.txt"
