import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_ollama import ChatOllama
from ollama import ResponseError

from agro_rag import resilience
from agro_rag.cache import RedisCache, get_cache
from agro_rag.db.session import get_session
from agro_rag.llm import DeterministicChatModel, get_llm
from agro_rag.main import create_app
from agro_rag.observability import BREAKER_REJECTIONS, BREAKER_STATE, CACHE_REQUESTS
from agro_rag.resilience import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    ResilientTransport,
    RetryPolicy,
    breaker_states,
    get_breaker,
    resilient_call,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Sleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _flaky(failures: int, exc: Exception) -> Any:
    calls = {"n": 0}

    async def func() -> str:
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return "ok"

    func.calls = calls  # type: ignore[attr-defined]
    return func


# ---- retry policy ---------------------------------------------------------------------------


def test_backoff_grows_exponentially_and_is_capped() -> None:
    policy = RetryPolicy(attempts=5, base_delay=0.1, max_delay=0.5)

    assert [policy.delay(n, rng=lambda: 1.0) for n in range(5)] == [0.1, 0.2, 0.4, 0.5, 0.5]


def test_backoff_applies_full_jitter() -> None:
    policy = RetryPolicy(base_delay=0.2, max_delay=5)

    assert policy.delay(2, rng=lambda: 0.0) == 0.0
    assert policy.delay(2, rng=lambda: 0.5) == pytest.approx(0.4)
    assert 0.0 <= policy.delay(3) <= 1.6


async def test_retries_until_the_call_succeeds() -> None:
    breaker = CircuitBreaker("t-retry", failure_threshold=10)
    sleeper = Sleeper()
    func = _flaky(2, ConnectionError("boom"))

    result = await resilient_call(
        func,
        breaker=breaker,
        policy=RetryPolicy(attempts=3, base_delay=0.1),
        retry_on=(ConnectionError,),
        sleep=sleeper,
        rng=lambda: 1.0,
    )

    assert result == "ok"
    assert func.calls["n"] == 3
    assert sleeper.delays == [0.1, 0.2]
    assert breaker.state is BreakerState.CLOSED


async def test_raises_the_last_error_after_the_attempts_are_exhausted() -> None:
    breaker = CircuitBreaker("t-exhaust", failure_threshold=10)
    func = _flaky(99, ConnectionError("boom"))

    with pytest.raises(ConnectionError):
        await resilient_call(
            func,
            breaker=breaker,
            policy=RetryPolicy(attempts=3),
            retry_on=(ConnectionError,),
            sleep=Sleeper(),
        )

    assert func.calls["n"] == 3


async def test_errors_outside_retry_on_propagate_without_retry_or_breaker_count() -> None:
    breaker = CircuitBreaker("t-other", failure_threshold=1)
    func = _flaky(99, ValueError("bug"))

    with pytest.raises(ValueError, match="bug"):
        await resilient_call(
            func,
            breaker=breaker,
            policy=RetryPolicy(attempts=3),
            retry_on=(ConnectionError,),
            sleep=Sleeper(),
        )

    assert func.calls["n"] == 1
    assert breaker.state is BreakerState.CLOSED


async def test_retries_stop_as_soon_as_the_circuit_opens() -> None:
    breaker = CircuitBreaker("t-stop", failure_threshold=2)
    func = _flaky(99, ConnectionError("boom"))

    with pytest.raises(CircuitOpenError):
        await resilient_call(
            func,
            breaker=breaker,
            policy=RetryPolicy(attempts=5),
            retry_on=(ConnectionError,),
            sleep=Sleeper(),
        )

    assert func.calls["n"] == 2
    assert breaker.state is BreakerState.OPEN


# ---- circuit breaker ------------------------------------------------------------------------


def test_breaker_opens_after_consecutive_failures_and_rejects_calls() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("t-open", failure_threshold=3, recovery_timeout=30, clock=clock)
    rejections = BREAKER_REJECTIONS.labels("t-open")._value.get()

    for _ in range(3):
        breaker.before_call()
        breaker.record_failure()

    assert breaker.state is BreakerState.OPEN
    assert BREAKER_STATE.labels("t-open")._value.get() == BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()
    assert BREAKER_REJECTIONS.labels("t-open")._value.get() == rejections + 1


def test_a_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker("t-reset", failure_threshold=3)

    for _ in range(2):
        breaker.record_failure()
    breaker.record_success()
    for _ in range(2):
        breaker.record_failure()

    assert breaker.state is BreakerState.CLOSED


def test_breaker_lets_one_probe_through_after_the_recovery_timeout() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("t-half", failure_threshold=1, recovery_timeout=30, clock=clock)
    breaker.record_failure()

    clock.now += 29
    with pytest.raises(CircuitOpenError):
        breaker.before_call()

    clock.now += 2
    breaker.before_call()
    assert breaker.state is BreakerState.HALF_OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()  # only one probe at a time


def test_reported_state_shows_half_open_once_the_timeout_has_passed() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("t-report", failure_threshold=1, recovery_timeout=30, clock=clock)
    breaker.record_failure()
    assert breaker.reported_state.name == "OPEN"

    clock.now += 31

    assert breaker.reported_state.name == "HALF_OPEN"
    assert breaker.state.name == "OPEN"


def test_successful_probe_closes_the_circuit() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("t-close", failure_threshold=1, recovery_timeout=30, clock=clock)
    breaker.record_failure()
    clock.now += 31

    breaker.before_call()
    breaker.record_success()

    assert breaker.state is BreakerState.CLOSED
    breaker.before_call()


def test_failed_probe_reopens_the_circuit_for_another_full_timeout() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("t-reopen", failure_threshold=5, recovery_timeout=30, clock=clock)
    for _ in range(5):
        breaker.record_failure()
    clock.now += 31

    breaker.before_call()
    breaker.record_failure()

    assert breaker.state is BreakerState.OPEN
    clock.now += 29
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_registry_returns_the_same_breaker_per_name() -> None:
    first = get_breaker("t-registry", failure_threshold=2, recovery_timeout=5)

    assert get_breaker("t-registry", failure_threshold=9, recovery_timeout=9) is first
    assert breaker_states()["t-registry"] == "closed"
    assert resilience.CircuitOpenError("x").name == "x"


# ---- HTTP transport (Ollama) ----------------------------------------------------------------


def _transport(
    handler: Any, *, failures: int = 5, attempts: int = 3
) -> tuple[ResilientTransport, CircuitBreaker]:
    breaker = CircuitBreaker("t-http", failure_threshold=failures)
    transport = ResilientTransport(
        httpx.MockTransport(handler),
        breaker,
        RetryPolicy(attempts=attempts),
        sleep=Sleeper(),
    )
    return transport, breaker


async def test_transport_retries_transient_5xx_then_succeeds() -> None:
    statuses = iter([503, 502, 200])
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        seen.append(status)
        return httpx.Response(status, json={"ok": True})

    transport, breaker = _transport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("http://ollama/api/tags")

    assert response.status_code == 200
    assert seen == [503, 502, 200]
    assert breaker.state is BreakerState.CLOSED


async def test_transport_retries_connection_errors_then_gives_up() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    transport, _ = _transport(handler, attempts=3)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://ollama/api/tags")

    assert calls == 3


async def test_transport_raises_a_status_error_when_5xx_persists() -> None:
    transport, _ = _transport(lambda request: httpx.Response(503), attempts=2)

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError, match="503 after 2 attempts"):
            await client.get("http://ollama/api/tags")


async def test_transport_does_not_retry_client_or_server_errors_it_cannot_fix() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500 if calls == 1 else 404)

    transport, breaker = _transport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await client.get("http://ollama/x")).status_code == 500
        assert (await client.get("http://ollama/x")).status_code == 404

    assert calls == 2
    assert breaker.state is BreakerState.CLOSED


async def test_transport_read_timeout_is_not_retried_but_counts_for_the_breaker() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    transport, breaker = _transport(handler, failures=2)
    async with httpx.AsyncClient(transport=transport) as client:
        for _ in range(2):
            with pytest.raises(httpx.ReadTimeout):
                await client.get("http://ollama/x")

    assert calls == 2
    assert breaker.state is BreakerState.OPEN


async def test_open_circuit_rejects_requests_without_reaching_ollama() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    transport, breaker = _transport(handler, failures=3, attempts=3)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://ollama/x")
        assert breaker.state is BreakerState.OPEN
        reached = calls
        with pytest.raises(CircuitOpenError):
            await client.get("http://ollama/x")

    assert reached == 3
    assert calls == 3


def _chat_line(content: str) -> str:
    return json.dumps(
        {
            "model": "m",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
        }
    )


async def test_chat_ollama_recovers_from_transient_failures() -> None:
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        if next(statuses) == 503:
            return httpx.Response(503)
        return httpx.Response(200, content=_chat_line("olá") + "\n")

    transport, _ = _transport(handler)
    llm = ChatOllama(model="m", async_client_kwargs={"transport": transport})

    assert (await llm.ainvoke("oi")).content == "olá"


async def test_chat_ollama_surfaces_a_persistent_outage() -> None:
    transport, _ = _transport(lambda request: httpx.Response(503), attempts=2)
    llm = ChatOllama(model="m", async_client_kwargs={"transport": transport})

    with pytest.raises((ResponseError, httpx.HTTPStatusError)):
        await llm.ainvoke("oi")


def test_get_llm_wires_timeouts_and_the_resilient_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGRO_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGRO_OLLAMA_TIMEOUT_SECONDS", "42")

    llm = get_llm()

    assert isinstance(llm, ChatOllama)
    async_http = llm._async_client._client
    assert async_http.timeout.read == 42
    assert async_http.timeout.connect == 3.0
    assert "ollama" in breaker_states()


# ---- Redis behind the breaker ---------------------------------------------------------------


class _CountingDownRedis:
    def __init__(self) -> None:
        self.calls = 0

    async def get(self, *_: object) -> None:
        self.calls += 1
        raise ConnectionError("down")

    async def set(self, *_: object, **__: object) -> None:
        self.calls += 1
        raise ConnectionError("down")


async def test_redis_retries_then_the_breaker_stops_calling_a_dead_server() -> None:
    client = _CountingDownRedis()
    breaker = CircuitBreaker("t-redis", failure_threshold=3, recovery_timeout=30)
    cache = RedisCache(
        client,  # type: ignore[arg-type]
        breaker=breaker,
        policy=RetryPolicy(attempts=2, base_delay=0, max_delay=0),
    )
    skipped = CACHE_REQUESTS.labels("unit", "skipped")._value.get()
    errors = CACHE_REQUESTS.labels("unit", "error")._value.get()

    assert await cache.get("unit", "k") is None  # 2 attempts
    assert await cache.get("unit", "k") is None  # 1 attempt opens the circuit, retry rejected
    assert breaker.state is BreakerState.OPEN
    calls = client.calls
    for _ in range(50):
        assert await cache.get("unit", "k") is None
        await cache.set("unit", "k", 1, ttl=1)

    assert calls == 3
    assert client.calls == 3
    assert CACHE_REQUESTS.labels("unit", "error")._value.get() == errors + 1
    assert CACHE_REQUESTS.labels("unit", "skipped")._value.get() == skipped + 101


# ---- probes and error mapping ---------------------------------------------------------------


class _BrokenSession:
    async def execute(self, *_: object) -> None:
        raise ConnectionError("db down")


async def _broken_session() -> AsyncIterator[_BrokenSession]:
    yield _BrokenSession()


class _Down:
    async def ping(self) -> None:
        raise ConnectionError("down")


async def test_liveness_never_touches_dependencies(client: AsyncClient) -> None:
    response = await client.get("/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_liveness_survives_a_database_outage() -> None:
    app = create_app()
    app.dependency_overrides[get_session] = _broken_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/livez")

    assert response.status_code == 200


async def test_readiness_reports_dependencies() -> None:
    app = create_app()
    app.dependency_overrides[get_cache] = lambda: None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "version": "0.1.0",
        "database": "ok",
        "cache": "disabled",
        "llm": "fake",
    }


async def test_readiness_fails_only_when_the_database_is_down() -> None:
    app = create_app()
    app.dependency_overrides[get_session] = _broken_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


async def test_readiness_stays_ready_when_redis_or_ollama_are_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGRO_LLM_PROVIDER", "ollama")
    monkeypatch.setattr("agro_rag.api.routes.breaker_states", lambda: {"ollama": "open"})
    app = create_app()
    app.dependency_overrides[get_cache] = lambda: RedisCache(_Down())  # type: ignore[arg-type]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["cache"], body["llm"]) == ("ready", "unavailable", "unavailable")


class _ExplodingChat(DeterministicChatModel):
    error: Exception

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


@pytest.mark.parametrize(
    "error",
    [
        CircuitOpenError("ollama"),
        ConnectionError("Failed to connect to Ollama"),
        ResponseError("model not found", 404),
        httpx.ReadTimeout("slow"),
    ],
)
async def test_llm_outage_is_a_503_with_retry_after_and_search_keeps_working(
    error: Exception,
) -> None:
    app = create_app()
    app.dependency_overrides[get_llm] = lambda: _ExplodingChat(error=error)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        ask = await c.post("/ask", json={"question": "como corrigir a acidez do solo?"})
        agent = await c.post("/agent", json={"question": "como corrigir a acidez do solo?"})
        search = await c.post("/search", json={"query": "acidez do solo"})

    assert ask.status_code == agent.status_code == 503
    assert ask.headers["retry-after"] == "30"
    assert "unavailable" in ask.json()["detail"]
    assert search.status_code == 200
