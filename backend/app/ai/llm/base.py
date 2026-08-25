from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMProvider(ABC):
    model_name: str

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...

    @abstractmethod
    def stream_generate(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """Yields answer text incrementally as the model generates it."""
        ...
