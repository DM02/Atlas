from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.models.user import ROLE_ADMIN, ROLE_USER, User
from tests.integration.factories import create_user


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> Generator[None]:
    """The rate limiter is a module-level singleton (in-memory, per-process —
    see core/rate_limit.py), so without this its state leaks across tests in
    the same pytest run: enough tests hitting /auth/register trip the real
    5/minute limit and later tests see 429s that have nothing to do with what
    they're testing.
    """
    limiter.reset()
    yield


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession]:
    """A session bound to a single connection + outer transaction that's rolled
    back after the test, so integration tests never leave rows behind — even
    though the code under test calls session.commit() internally (that becomes
    a SAVEPOINT release, not a real commit, via join_transaction_mode).

    Uses its own engine (NullPool, disposed at teardown) rather than the app's
    module-level engine: pytest-asyncio gives each test its own event loop, and
    a pooled asyncpg connection created in one loop breaks when cleaned up from
    another — this scopes the engine's lifetime to the test's loop instead.
    """
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            session = AsyncSession(
                bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
            try:
                yield session
            finally:
                await session.close()
                await trans.rollback()
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def admin_user(db_session: AsyncSession) -> User:
    return await create_user(db_session, email="admin@example.com", role_name=ROLE_ADMIN)


@pytest_asyncio.fixture
async def regular_user(db_session: AsyncSession) -> User:
    return await create_user(db_session, email="user@example.com", role_name=ROLE_USER)
