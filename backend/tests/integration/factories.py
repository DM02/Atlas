import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings.base import EmbeddingProvider
from app.core.config import Settings
from app.core.security import hash_password
from app.core.storage import StorageBackend
from app.models.document import Document
from app.models.user import Role, User
from app.services.ingestion_service import create_pending_version, process_document_version

TEST_PASSWORD = "correct horse battery staple"


async def create_user(session: AsyncSession, *, email: str, role_name: str) -> User:
    """Persists a user directly (bypassing the HTTP register flow) with a known
    role and password — for tests that need a user to exist but aren't testing
    registration itself. Roles are seeded by the auth migration, so this expects
    role_name to already exist.
    """
    role = (await session.execute(select(Role).where(Role.name == role_name))).scalar_one()
    user = User(email=email, hashed_password=hash_password(TEST_PASSWORD), role_id=role.id)
    session.add(user)
    await session.flush()
    await session.refresh(user, attribute_names=["role"])
    return user


async def ingest_test_document(
    *,
    session: AsyncSession,
    storage: StorageBackend,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    filename: str,
    content: bytes,
    owner_id: uuid.UUID,
    is_private: bool = False,
    document: Document | None = None,
) -> Document:
    """Test convenience: runs both halves of the Phase 4 pipeline synchronously
    (create_pending_version + process_document_version) in one call, mirroring
    what the API + Arq worker do in production across two processes. Most tests
    don't care about the async boundary, just the end result.
    """
    version = await create_pending_version(
        session=session,
        storage=storage,
        settings=settings,
        filename=filename,
        content=content,
        owner_id=owner_id,
        document=document,
        is_private=is_private,
    )
    await process_document_version(
        session=session,
        storage=storage,
        embedding_provider=embedding_provider,
        settings=settings,
        document_version_id=version.id,
    )
    result = await session.get(Document, version.document_id)
    assert result is not None
    return result


class ImmediateArqPool:
    """Fake Arq pool: runs the enqueued job function synchronously instead of
    pushing it over Redis, so API-level tests can exercise the real
    create_pending_version/process_document_version split without a live
    worker process or Redis connection.
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        storage: StorageBackend,
        embedding_provider: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._session = session
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._settings = settings

    async def enqueue_job(self, function_name: str, *args: object) -> None:
        if function_name != "run_ingestion":
            raise ValueError(f"ImmediateArqPool does not know how to run {function_name!r}")
        (document_version_id,) = args
        await process_document_version(
            session=self._session,
            storage=self._storage,
            embedding_provider=self._embedding_provider,
            settings=self._settings,
            document_version_id=uuid.UUID(document_version_id),
        )
