from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings


class StorageBackend(ABC):
    """Content-addressable-ish blob storage for raw uploaded documents.

    Swappable for an S3/MinIO-backed implementation later without touching callers.
    """

    @abstractmethod
    async def save(self, key: str, content: bytes) -> str:
        """Persist content under key, returning a backend-specific location string."""

    @abstractmethod
    async def read(self, key: str) -> bytes: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def save(self, key: str, content: bytes) -> str:
        path = self._base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    async def read(self, key: str) -> bytes:
        return (self._base_path / key).read_bytes()

    async def delete(self, key: str) -> None:
        (self._base_path / key).unlink(missing_ok=True)


@lru_cache
def get_storage_backend() -> StorageBackend:
    return LocalStorageBackend(get_settings().storage_path)
