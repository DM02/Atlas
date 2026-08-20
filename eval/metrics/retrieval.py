"""Retrieval-quality metrics: Recall@K, Precision@K, MRR.

Ground truth in golden_qa.yaml is at DOCUMENT granularity (see its schema
comment), so these functions are generic over any hashable ID — callers pass
document identifiers (e.g. source filenames), not chunk IDs, so results stay
comparable across chunking strategies with different chunk boundaries.

Callers should only pass ANSWERABLE golden questions into these functions —
"how well did retrieval find the right document" isn't a meaningful question
for questions with no right document. Unanswerable questions are instead
scored via eval/metrics/generation.py's correct_refusal_rate/hallucination_rate.
"""

from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents present in the top k retrieved."""
    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    found = len(set(top_k) & relevant)
    return found / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k retrieved documents that are actually relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    found = len(set(top_k) & relevant)
    return found / len(top_k)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document found; 0.0 if none is present."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: list[float]) -> float:
    """Mean of per-query reciprocal_rank() values across a query set."""
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
