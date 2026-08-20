from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
