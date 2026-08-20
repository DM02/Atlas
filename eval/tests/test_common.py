import numpy as np
import pytest

from eval.runners.common import (
    ZeroPaddedEmbeddingProvider,
    dedupe_preserve_order,
    load_corpus,
    load_golden_questions,
    rank_by_cosine_similarity,
)


class FakeEmbeddingProvider:
    model_name = "fake-embed"

    def __init__(self, dimension: int, vectors: dict[str, list[float]] | None = None) -> None:
        self.dimension = dimension
        self._vectors = vectors or {}

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(text, [0.1] * self.dimension) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def test_load_golden_questions_has_expected_answerable_split() -> None:
    questions = load_golden_questions()

    assert len(questions) == 30
    assert sum(1 for q in questions if q.answerable) == 24
    assert sum(1 for q in questions if not q.answerable) == 6
    assert len({q.id for q in questions}) == 30  # unique ids


def test_load_corpus_loads_all_four_documents_with_pages() -> None:
    corpus = load_corpus()

    filenames = {doc.filename for doc in corpus}
    assert filenames == {
        "employee_handbook.txt",
        "engineering_onboarding.txt",
        "product_overview.txt",
        "security_policy.txt",
    }
    assert all(doc.pages for doc in corpus)
    assert all(doc.pages[0].text for doc in corpus)


def test_dedupe_preserve_order_keeps_first_occurrence_rank() -> None:
    result = dedupe_preserve_order(["b.txt", "a.txt", "b.txt", "c.txt", "a.txt"])

    assert result == ["b.txt", "a.txt", "c.txt"]


def test_dedupe_preserve_order_empty() -> None:
    assert dedupe_preserve_order([]) == []


def test_rank_by_cosine_similarity_orders_by_closeness() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array(
        [
            [0.0, 1.0],  # orthogonal, least similar
            [1.0, 0.0],  # identical, most similar
            [0.7, 0.7],  # partially similar
        ],
        dtype=np.float32,
    )

    ranked = rank_by_cosine_similarity(query, candidates)

    assert list(ranked) == [1, 2, 0]


def test_rank_by_cosine_similarity_ignores_magnitude() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[100.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    ranked = rank_by_cosine_similarity(query, candidates)

    assert ranked[0] == 0  # same direction as query, despite larger magnitude


async def test_zero_padded_embedding_provider_pads_to_target_dimension() -> None:
    inner = FakeEmbeddingProvider(dimension=4)
    provider = ZeroPaddedEmbeddingProvider(inner, target_dimension=8)

    [vector] = await provider.embed_documents(["hello"])

    assert len(vector) == 8
    assert vector[4:] == [0.0, 0.0, 0.0, 0.0]
    assert provider.dimension == 8


async def test_zero_padded_embedding_provider_preserves_cosine_similarity() -> None:
    vectors = {"a": [1.0, 2.0, 3.0], "b": [4.0, 1.0, 0.0]}
    inner = FakeEmbeddingProvider(dimension=3, vectors=vectors)
    provider = ZeroPaddedEmbeddingProvider(inner, target_dimension=6)

    raw_a, raw_b = await inner.embed_documents(["a", "b"])
    padded_a, padded_b = await provider.embed_documents(["a", "b"])

    def cosine(u: list[float], v: list[float]) -> float:
        u_arr, v_arr = np.array(u), np.array(v)
        return float(u_arr @ v_arr / (np.linalg.norm(u_arr) * np.linalg.norm(v_arr)))

    assert cosine(padded_a, padded_b) == pytest.approx(cosine(raw_a, raw_b))


def test_zero_padded_embedding_provider_rejects_smaller_target() -> None:
    inner = FakeEmbeddingProvider(dimension=8)
    with pytest.raises(ValueError, match="target_dimension"):
        ZeroPaddedEmbeddingProvider(inner, target_dimension=4)


async def test_zero_padded_embedding_provider_noop_when_dimensions_equal() -> None:
    inner = FakeEmbeddingProvider(dimension=4, vectors={"x": [1.0, 2.0, 3.0, 4.0]})
    provider = ZeroPaddedEmbeddingProvider(inner, target_dimension=4)

    [vector] = await provider.embed_documents(["x"])

    assert vector == [1.0, 2.0, 3.0, 4.0]
