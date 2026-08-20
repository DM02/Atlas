from functools import lru_cache

from app.ai.llm.base import LLMProvider
from app.ai.llm.openai import OpenAILLMProvider
from app.core.config import get_settings


@lru_cache
def get_llm_provider() -> LLMProvider:
    """When llm_base_url is set (default: OpenRouter), the LLM key/endpoint
    are independent of the embedding provider's OpenAI key — see
    core/config.py's llm_base_url comment for why.
    """
    settings = get_settings()
    if settings.llm_base_url:
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return OpenAILLMProvider(
            api_key=settings.openrouter_api_key,
            model_name=settings.llm_model,
            base_url=settings.llm_base_url,
        )

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAILLMProvider(api_key=settings.openai_api_key, model_name=settings.llm_model)
