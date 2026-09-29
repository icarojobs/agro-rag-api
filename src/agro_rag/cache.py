import hashlib
import json
from functools import lru_cache
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from agro_rag.config import get_settings
from agro_rag.observability import CACHE_REQUESTS

logger = structlog.get_logger(__name__)

_PREFIX = "agro"
_SCAN_COUNT = 200


def make_key(cache: str, *parts: object, scope: str = "") -> str:
    """Build `agro:<cache>[:<scope>]:<sha256>` so a whole scope can be invalidated by pattern."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return ":".join(p for p in (_PREFIX, cache, scope, digest) if p)


class RedisCache:
    """Cache-aside helper that never raises: a Redis failure is just a cache miss."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, cache: str, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
            value = None if raw is None else json.loads(raw)
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            CACHE_REQUESTS.labels(cache, "error").inc()
            logger.warning("cache_get_failed", cache=cache, error=type(exc).__name__)
            return None
        CACHE_REQUESTS.labels(cache, "miss" if value is None else "hit").inc()
        return value

    async def set(self, cache: str, key: str, value: Any, ttl: int) -> None:
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
        except (RedisError, OSError, TimeoutError) as exc:
            CACHE_REQUESTS.labels(cache, "error").inc()
            logger.warning("cache_set_failed", cache=cache, error=type(exc).__name__)

    async def delete_matching(self, pattern: str) -> int:
        """Delete keys by glob pattern with SCAN (non-blocking). Returns how many were removed."""
        removed = 0
        try:
            async for key in self._client.scan_iter(match=pattern, count=_SCAN_COUNT):
                removed += await self._client.delete(key)
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("cache_invalidate_failed", pattern=pattern, error=type(exc).__name__)
        return removed

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except (RedisError, OSError, TimeoutError):
            return False

    async def close(self) -> None:
        await self._client.aclose()


@lru_cache
def get_cache() -> RedisCache | None:
    settings = get_settings()
    if not settings.redis_url:
        return None
    client: Redis = Redis.from_url(
        settings.redis_url,
        socket_timeout=settings.redis_timeout_seconds,
        socket_connect_timeout=settings.redis_timeout_seconds,
        decode_responses=True,
    )
    return RedisCache(client)


async def close_cache() -> None:
    cache = get_cache()
    if cache is not None:
        await cache.close()
    get_cache.cache_clear()


async def invalidate_collection(collection: str) -> None:
    """Drop cached results derived from a collection after its content changed."""
    cache = get_cache()
    if cache is None:
        return
    removed = await cache.delete_matching(f"{_PREFIX}:search:{collection}:*")
    removed += await cache.delete_matching(f"{_PREFIX}:answer:{collection}:*")
    logger.info("cache_invalidated", collection=collection, keys=removed)
