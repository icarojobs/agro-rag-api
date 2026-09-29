import hashlib
import json
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from agro_rag.config import get_settings
from agro_rag.observability import CACHE_REQUESTS
from agro_rag.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryPolicy,
    get_breaker,
    resilient_call,
)

logger = structlog.get_logger(__name__)

_PREFIX = "agro"
_SCAN_COUNT = 200
_RETRY_ON: tuple[type[BaseException], ...] = (RedisError, OSError, TimeoutError)


def make_key(cache: str, *parts: object, scope: str = "") -> str:
    """Build `agro:<cache>[:<scope>]:<sha256>` so a whole scope can be invalidated by pattern."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return ":".join(p for p in (_PREFIX, cache, scope, digest) if p)


class RedisCache:
    """Cache-aside helper that never raises: a Redis failure is just a cache miss.

    Each operation is retried with backoff and goes through a circuit breaker, so a dead
    Redis stops costing a timeout per request once the circuit opens.
    """

    def __init__(
        self,
        client: Redis,
        *,
        breaker: CircuitBreaker | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        self._client = client
        self._breaker = breaker or CircuitBreaker("redis", failure_threshold=3, recovery_timeout=10)
        self._policy = policy or RetryPolicy(attempts=2, base_delay=0.02, max_delay=0.1)

    async def _call[T](self, func: Callable[[], Awaitable[T]]) -> T:
        return await resilient_call(
            func, breaker=self._breaker, policy=self._policy, retry_on=_RETRY_ON
        )

    async def get(self, cache: str, key: str) -> Any | None:
        try:
            raw = await self._call(lambda: self._client.get(key))
            value = None if raw is None else json.loads(raw)
        except CircuitOpenError:
            CACHE_REQUESTS.labels(cache, "skipped").inc()
            return None
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            CACHE_REQUESTS.labels(cache, "error").inc()
            logger.warning("cache_get_failed", cache=cache, error=type(exc).__name__)
            return None
        CACHE_REQUESTS.labels(cache, "miss" if value is None else "hit").inc()
        return value

    async def set(self, cache: str, key: str, value: Any, ttl: int) -> None:
        payload = json.dumps(value)
        try:
            await self._call(lambda: self._client.set(key, payload, ex=ttl))
        except CircuitOpenError:
            CACHE_REQUESTS.labels(cache, "skipped").inc()
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
        """Direct probe, outside the breaker, so readiness reflects the real state."""
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
    breaker = get_breaker(
        "redis",
        failure_threshold=settings.redis_breaker_failures,
        recovery_timeout=settings.redis_breaker_recovery_seconds,
    )
    policy = RetryPolicy(
        attempts=settings.redis_retry_attempts,
        base_delay=settings.redis_retry_base_delay_seconds,
        max_delay=0.25,
    )
    return RedisCache(client, breaker=breaker, policy=policy)


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
