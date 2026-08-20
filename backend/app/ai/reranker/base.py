from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankCandidate:
    id: str
    text: str


class RerankerProvider(ABC):
    model_name: str

    @abstractmethod
    async def rerank(
        self, query: str, candidates: list[RerankCandidate], top_n: int
    ) -> list[tuple[str, float]]:
        """Return (candidate_id, score) pairs, best-first, length <= top_n."""
