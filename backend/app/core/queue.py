import asyncio

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.core.config import get_settings

_pool: ArqRedis | None = None
_pool_lock = asyncio.Lock()


async def get_arq_pool() -> ArqRedis:
    """Lazy async singleton for the Arq/Redis connection pool used to enqueue
    ingestion jobs from FastAPI request handlers. Can't use functools.lru_cache
    here (it caches a coroutine object, not its result) — the lock makes
    first-creation safe if two requests race before the pool exists yet.
    """
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _pool
