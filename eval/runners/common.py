"""Shared loading/embedding/ranking helpers used by both experiment runners
(eval/runners/in_memory_experiments.py, eval/runners/pgvector_experiments.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.pipeline.extraction import MIME_TXT, ExtractedPage, extract_text

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "eval" / "dataset"
CORPUS_DIR = DATASET_DIR / "corpus"
GOLDEN_QA_PATH = DATASET_DIR / "golden_qa.yaml"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"


@dataclass
class GoldenQuestion:
    id: str
    question: str
    answerable: bool
    expected_answer_contains: list[str]
    expected_source_documents: list[str]
    expected_source_sections: list[str]


@dataclass
class CorpusDocument:
    filename: str
    pages: list[ExtractedPage]


def load_golden_questions(path: Path = GOLDEN_QA_PATH) -> list[GoldenQuestion]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [
        GoldenQuestion(
            id=entry["id"],
            question=entry["question"],
            answerable=entry["answerable"],
            expected_answer_contains=entry["expected_answer_contains"],
            expected_source_documents=entry["expected_source_documents"],
            expected_source_sections=entry["expected_source_sections"],
        )
        for entry in raw
    ]


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[CorpusDocument]:
    documents = []
    for path in sorted(corpus_dir.glob("*.txt")):
        content = path.read_bytes()
        documents.append(CorpusDocument(filename=path.name, pages=extract_text(content, MIME_TXT)))
    return documents


async def embed_all(provider: EmbeddingProvider, texts: list[str]) -> np.ndarray:
    vectors = await provider.embed_documents(texts)
    return np.array(vectors, dtype=np.float32)


def rank_by_cosine_similarity(
    query_vector: np.ndarray, candidate_vectors: np.ndarray
) -> np.ndarray:
    """Returns candidate indices sorted by descending cosine similarity to the query."""
    query_norm = query_vector / np.linalg.norm(query_vector)
    candidate_norms = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    scores = candidate_norms @ query_norm
    return np.argsort(-scores)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def write_report(name: str, payload: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


class ZeroPaddedEmbeddingProvider(EmbeddingProvider):
    """Wraps a smaller-dimension local EmbeddingProvider and zero-pads its
    output vectors up to `target_dimension`.

    Zero-padding provably preserves cosine similarity exactly: padded zeros
    contribute nothing to the dot product, and appending zeros doesn't change
    a vector's norm either, so cosine_similarity(pad(a), pad(b)) ==
    cosine_similarity(a, b) for any pair. This exists only to satisfy
    DocumentChunk.embedding's fixed Vector(1536) column (sized for OpenAI's
    text-embedding-3-small — see docs/ARCHITECTURE.md §10 risk 10.2), so the
    REAL pgvector-backed retrieval_service can be exercised for the
    hybrid-search/reranking comparison (eval/runners/pgvector_experiments.py)
    without live OpenAI access, which this project's account doesn't have.
    It does not change
    retrieval ranking versus using the inner provider's vectors directly.
    """

    def __init__(self, inner: EmbeddingProvider, target_dimension: int) -> None:
        if target_dimension < inner.dimension:
            raise ValueError("target_dimension must be >= the inner provider's dimension")
        self._inner = inner
        self._pad_width = target_dimension - inner.dimension
        self.model_name = f"{inner.model_name}+zero-padded-to-{target_dimension}"
        self.dimension = target_dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await self._inner.embed_documents(texts)
        if self._pad_width == 0:
            return vectors
        padding = [0.0] * self._pad_width
        return [vector + padding for vector in vectors]
