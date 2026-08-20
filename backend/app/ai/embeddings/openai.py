from openai import AsyncOpenAI, OpenAIError

from app.ai.embeddings.base import EmbeddingProvider
from app.ai.errors import AIProviderError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Talks to any OpenAI-Embeddings-compatible API — real OpenAI when
    base_url is None, or a gateway like OpenRouter when it's set (see
    core/config.py's embedding_base_url).
    """

    def __init__(
        self, api_key: str, model_name: str, dimension: int, base_url: str | None = None
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(model=self.model_name, input=texts)
        except OpenAIError as exc:
            raise AIProviderError(f"Embedding request failed: {exc}") from exc
        return [item.embedding for item in response.data]
