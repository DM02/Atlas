from eval.metrics.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_finds_all_relevant_within_k() -> None:
    retrieved = ["a.txt", "b.txt", "c.txt"]
    relevant = {"b.txt"}

    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_misses_relevant_beyond_k() -> None:
    retrieved = ["a.txt", "b.txt", "c.txt"]
    relevant = {"c.txt"}

    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k_partial_when_only_some_relevant_found() -> None:
    retrieved = ["a.txt", "b.txt"]
    relevant = {"a.txt", "z.txt"}

    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k_empty_relevant_set_is_trivially_satisfied() -> None:
    assert recall_at_k(["a.txt"], set(), k=1) == 1.0


def test_precision_at_k_all_relevant() -> None:
    retrieved = ["a.txt", "b.txt"]
    relevant = {"a.txt", "b.txt"}

    assert precision_at_k(retrieved, relevant, k=2) == 1.0


def test_precision_at_k_none_relevant() -> None:
    retrieved = ["a.txt", "b.txt"]
    relevant = {"z.txt"}

    assert precision_at_k(retrieved, relevant, k=2) == 0.0


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], {"a.txt"}, k=5) == 0.0


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(["a.txt", "b.txt"], {"a.txt"}) == 1.0


def test_reciprocal_rank_third_position() -> None:
    assert reciprocal_rank(["x.txt", "y.txt", "a.txt"], {"a.txt"}) == 1 / 3


def test_reciprocal_rank_not_found() -> None:
    assert reciprocal_rank(["x.txt"], {"a.txt"}) == 0.0


def test_mean_reciprocal_rank_averages() -> None:
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == 0.5


def test_mean_reciprocal_rank_empty_is_zero() -> None:
    assert mean_reciprocal_rank([]) == 0.0
