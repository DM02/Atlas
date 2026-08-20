from abc import ABC, abstractmethod


class LLMProvider(ABC):
    model_name: str

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...
