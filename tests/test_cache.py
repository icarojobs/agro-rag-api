import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from httpx import ASGITransport, AsyncClient
from langchain_core.embeddings import Embeddings
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag import cache as cache_module
from agro_rag import retrieval
from agro_rag.cache import RedisCache, close_cache, get_cache, invalidate_collection, make_key
from agro_rag.config import get_settings
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.ingestion.pipeline import ingest_corpus
from agro_rag.main import create_app
from agro_rag.observability import CACHE_REQUESTS


class _DownRedis:
    """Stands in for a Redis server that refuses every connection."""

    async def get(self, *_: object) -> None:
        raise RedisConnectionError("refused")

    async def set(self, *_: object, **__: object) -> None:
        raise RedisConnectionError("refused")

    async def ping(self) -> None:
        raise RedisConnectionError("refused")

    def scan_iter(self, **_: object) -> AsyncIterator[str]:
        raise RedisConnectionError("refused")

    async def aclose(self) -> None:
        return None


class _CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.calls = 0
        self._inner = HashingEmbeddings()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._inner.embed_query(text)


def _count(cache: str, result: str) -> float:
    return float(CACHE_REQUESTS.labels(cache, result)._value.get())


@pytest.fixture
async def fake_redis() -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture
def cache(fake_redis: FakeAsyncRedis) -> RedisCache:
    return RedisCache(fake_redis)


@pytest.fixture
def down_cache() -> RedisCache:
    return RedisCache(_DownRedis())  # type: ignore[arg-type]


@pytest.fixture
async def indexed(session: AsyncSession, corpus_dir: Path) -> None:
    await ingest_corpus(
        session,
        HashingEmbeddings(),
        corpus_dir,
        collection="default",
        chunk_size=500,
        chunk_overlap=50,
    )


def _client_with(cache: RedisCache | None) -> AsyncClient:
    app = create_app()
    app.dependency_overrides[get_cache] = lambda: cache
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def test_make_key_is_stable_and_scoped() -> None:
    assert make_key("search", "a", 1, scope="c") == make_key("search", "a", 1, scope="c")
    assert make_key("search", "a", 1) != make_key("search", "a", 2)
    assert make_key("search", "a", scope="solo").startswith("agro:search:solo:")
    assert make_key("embedding", "a").startswith("agro:embedding:")


async def test_get_set_roundtrip_counts_hits_and_misses(cache: RedisCache) -> None:
    hits, misses = _count("unit", "hit"), _count("unit", "miss")

    assert await cache.get("unit", "k") is None
    await cache.set("unit", "k", {"a": [1, 2]}, ttl=30)
    assert await cache.get("unit", "k") == {"a": [1, 2]}

    assert _count("unit", "miss") == misses + 1
    assert _count("unit", "hit") == hits + 1


async def test_entries_expire_with_ttl(cache: RedisCache, fake_redis: FakeAsyncRedis) -> None:
    await cache.set("unit", "k", 1, ttl=30)

    assert 0 < await fake_redis.ttl("k") <= 30


async def test_corrupt_entry_is_treated_as_error(
    cache: RedisCache, fake_redis: FakeAsyncRedis
) -> None:
    await fake_redis.set("k", "{not json")
    errors = _count("unit", "error")

    assert await cache.get("unit", "k") is None
    assert _count("unit", "error") == errors + 1


async def test_redis_failures_never_raise(down_cache: RedisCache) -> None:
    errors = _count("unit", "error")

    assert await down_cache.get("unit", "k") is None
    await down_cache.set("unit", "k", 1, ttl=30)
    assert await down_cache.delete_matching("agro:*") == 0
    assert await down_cache.ping() is False
    assert _count("unit", "error") == errors + 2


async def test_delete_matching_only_removes_the_pattern(cache: RedisCache) -> None:
    await cache.set("unit", "agro:search:a:1", 1, ttl=30)
    await cache.set("unit", "agro:search:a:2", 1, ttl=30)
    await cache.set("unit", "agro:search:b:1", 1, ttl=30)

    assert await cache.delete_matching("agro:search:a:*") == 2
    assert await cache.get("unit", "agro:search:b:1") == 1
    assert await cache.ping() is True


@pytest.mark.usefixtures("indexed")
async def test_search_is_served_from_cache_on_second_call(
    session: AsyncSession, cache: RedisCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = retrieval.search_by_vector

    async def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(retrieval, "search_by_vector", counting)
    embeddings = _CountingEmbeddings()

    first = await retrieval.search(
        session, embeddings, "calcário acidez", k=2, collection="default", cache=cache
    )
    second = await retrieval.search(
        session, embeddings, "calcário acidez", k=2, collection="default", cache=cache
    )

    assert first == second
    assert (calls, embeddings.calls) == (1, 1)


@pytest.mark.usefixtures("indexed")
async def test_query_embedding_is_reused_across_different_filters(
    session: AsyncSession, cache: RedisCache
) -> None:
    embeddings = _CountingEmbeddings()

    await retrieval.search(session, embeddings, "calcário", k=1, collection="default", cache=cache)
    await retrieval.search(session, embeddings, "calcário", k=2, collection="default", cache=cache)

    assert embeddings.calls == 1


@pytest.mark.usefixtures("indexed")
async def test_search_falls_back_to_database_when_redis_is_down(
    session: AsyncSession, down_cache: RedisCache
) -> None:
    results = await retrieval.search(
        session, HashingEmbeddings(), "calcário", k=1, collection="default", cache=down_cache
    )

    assert results[0].source == "solo/calagem.md"


@pytest.mark.usefixtures("indexed")
async def test_search_endpoint_hits_cache_and_survives_redis_outage(cache: RedisCache) -> None:
    body = {"query": "percevejo na soja", "k": 1}
    hits = _count("search", "hit")

    async with _client_with(cache) as client:
        first = await client.post("/search", json=body)
        second = await client.post("/search", json=body)
    async with _client_with(RedisCache(_DownRedis())) as client:  # type: ignore[arg-type]
        degraded = await client.post("/search", json=body)

    assert first.status_code == second.status_code == degraded.status_code == 200
    assert first.json()["results"] == second.json()["results"] == degraded.json()["results"]
    assert _count("search", "hit") == hits + 1


@pytest.mark.usefixtures("indexed")
async def test_ask_answer_is_cached(cache: RedisCache) -> None:
    body = {"question": "como o calcário corrige a acidez do solo?"}

    async with _client_with(cache) as client:
        first = await client.post("/ask", json=body)
        second = await client.post("/ask", json=body)

    assert first.status_code == second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"]
    assert first.json()["sources"] == second.json()["sources"]


@pytest.mark.usefixtures("indexed")
async def test_ask_survives_redis_outage() -> None:
    async with _client_with(RedisCache(_DownRedis())) as client:  # type: ignore[arg-type]
        response = await client.post("/ask", json={"question": "como corrigir a acidez do solo?"})

    assert response.status_code == 200


async def test_ingestion_invalidates_cached_results(
    session: AsyncSession,
    corpus_dir: Path,
    cache: RedisCache,
    fake_redis: FakeAsyncRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "get_cache", lambda: cache)
    stale = make_key("search", "default", "q", scope="default")
    await cache.set("search", stale, [], ttl=60)
    other = make_key("search", "other", "q", scope="other")
    await cache.set("search", other, [], ttl=60)

    await ingest_corpus(
        session,
        HashingEmbeddings(),
        corpus_dir,
        collection="default",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert await fake_redis.exists(stale) == 0
    assert await fake_redis.exists(other) == 1


async def test_invalidate_is_a_noop_without_redis() -> None:
    await invalidate_collection("default")


async def test_get_cache_follows_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRO_REDIS_URL")
    get_settings.cache_clear()
    get_cache.cache_clear()
    assert get_cache() is None

    monkeypatch.setenv("AGRO_REDIS_URL", "redis://localhost:6399/0")
    get_settings.cache_clear()
    get_cache.cache_clear()
    assert isinstance(get_cache(), RedisCache)
    first = get_cache()
    await close_cache()
    assert get_cache() is not first
    await close_cache()


@pytest.mark.skipif(not os.environ.get("AGRO_REDIS_URL"), reason="needs a Redis server")
async def test_real_redis_roundtrip() -> None:
    cache = get_cache()
    assert cache is not None
    try:
        await cache.set("integration", "agro:integration:k", {"ok": True}, ttl=30)
        assert await cache.get("integration", "agro:integration:k") == {"ok": True}
        assert await cache.delete_matching("agro:integration:*") == 1
        assert await cache.ping() is True
    finally:
        await close_cache()


async def test_unreachable_real_redis_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    client: Redis = Redis.from_url(
        "redis://127.0.0.1:1/0", socket_connect_timeout=0.2, decode_responses=True
    )
    cache = RedisCache(client)

    assert await cache.get("unit", "k") is None
    await cache.set("unit", "k", 1, ttl=30)
    assert await cache.ping() is False
    await cache.close()
