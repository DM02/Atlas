from functools import lru_cache

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.embeddings.openai import OpenAIEmbeddingProvider
from app.core.config import get_settings


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_base_url:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return OpenAIEmbeddingProvider(
            api_key=settings.openrouter_api_key,
            model_name=settings.embedding_model,
            dimension=settings.openai_embedding_dimension,
            base_url=settings.embedding_base_url,
        )

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
        dimension=settings.openai_embedding_dimension,
    )
