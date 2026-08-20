from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.ai.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local CPU embedding model via sentence-transformers — free, no API key,
    used as the second embedding model for Phase 5's required model-comparison
    experiment (see docs/EVALUATION.md) without depending on OpenAI credits.

    Same lazy-loading pattern as ai/reranker/cross_encoder.py: the heavy
    import and model weights only load on first actual use, not at construction.
    """

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = await asyncio.to_thread(self._load_model)
        # normalize_embeddings=True: BGE models are trained/recommended for
        # cosine similarity on normalized vectors — matches how retrieval_service
        # and the eval runner both compare embeddings (cosine, not raw dot product).
        embeddings = await asyncio.to_thread(
            model.encode, texts, normalize_embeddings=True
        )
        return [vector.tolist() for vector in embeddings]
