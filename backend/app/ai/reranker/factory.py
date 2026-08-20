from functools import lru_cache

from app.ai.reranker.base import RerankerProvider
from app.ai.reranker.cross_encoder import CrossEncoderReranker
from app.core.config import get_settings


@lru_cache
def get_reranker() -> RerankerProvider:
    return CrossEncoderReranker(model_name=get_settings().reranker_model)
