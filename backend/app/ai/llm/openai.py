from openai import AsyncOpenAI, OpenAIError

from app.ai.errors import AIProviderError
from app.ai.llm.base import LLMProvider


class OpenAILLMProvider(LLMProvider):
    """Talks to any OpenAI-Chat-Completions-compatible API — real OpenAI when
    base_url is None, or a gateway like OpenRouter when it's set (see
    core/config.py's llm_base_url). The OpenAI SDK itself doesn't care which
    server answers, only that the request/response shape matches.
    """

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None) -> None:
        self.model_name = model_name
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
        except OpenAIError as exc:
            raise AIProviderError(f"Chat completion request failed: {exc}") from exc
        return response.choices[0].message.content or ""
