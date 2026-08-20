from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.ai.reranker.base import RerankCandidate, RerankerProvider

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker(RerankerProvider):
    """Local CPU cross-encoder (default: BAAI/bge-reranker-base).

    The `sentence-transformers`/torch import and the model weights themselves are
    both loaded lazily on first actual `rerank()` call, not at construction —
    this class is cheap to instantiate via FastAPI's `Depends(get_reranker)` even
    when `settings.enable_reranking` is False, so disabled-by-default reranking
    doesn't pay startup cost for a feature nobody asked for on this request.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self, query: str, candidates: list[RerankCandidate], top_n: int
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        model = await asyncio.to_thread(self._load_model)
        pairs = [(query, c.text) for c in candidates]
        scores = await asyncio.to_thread(model.predict, pairs)

        ranked = sorted(
            zip((c.id for c in candidates), scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(candidate_id, float(score)) for candidate_id, score in ranked[:top_n]]
