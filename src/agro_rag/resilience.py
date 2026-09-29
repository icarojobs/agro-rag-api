"""Retry with exponential backoff and jitter, plus a circuit breaker.

Both are dependency-agnostic; `ResilientTransport` applies them to the HTTP
calls made to Ollama, and `RedisCache` uses them for cache operations.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum

import httpx
import structlog

from agro_rag.observability import (
    BREAKER_REJECTIONS,
    BREAKER_STATE,
    DEPENDENCY_RETRIES,
)

logger = structlog.get_logger(__name__)


class CircuitOpenError(Exception):
    """Raised without calling the dependency while its circuit is open."""

    def __init__(self, name: str) -> None:
        super().__init__(f"circuit breaker '{name}' is open")
        self.name = name


class BreakerState(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures and rejects calls fast.

    After `recovery_timeout` seconds one probe call is let through (half-open): if it
    succeeds the circuit closes, if it fails the circuit opens again.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state = BreakerState.CLOSED
        self._probe_in_flight = False
        BREAKER_STATE.labels(name).set(self._state)

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def reported_state(self) -> BreakerState:
        """State for probes and dashboards: an open circuit past its timeout is half-open."""
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            return BreakerState.HALF_OPEN
        return self._state

    def _transition(self, state: BreakerState) -> None:
        if state is not self._state:
            logger.warning("circuit_breaker_transition", name=self.name, state=state.name.lower())
        self._state = state
        BREAKER_STATE.labels(self.name).set(state)

    def before_call(self) -> None:
        if self._state is BreakerState.OPEN:
            if self._clock() - self._opened_at < self.recovery_timeout:
                BREAKER_REJECTIONS.labels(self.name).inc()
                raise CircuitOpenError(self.name)
            self._transition(BreakerState.HALF_OPEN)
        if self._state is BreakerState.HALF_OPEN:
            if self._probe_in_flight:
                BREAKER_REJECTIONS.labels(self.name).inc()
                raise CircuitOpenError(self.name)
            self._probe_in_flight = True

    def record_success(self) -> None:
        self._failures = 0
        self._probe_in_flight = False
        self._transition(BreakerState.CLOSED)

    def record_failure(self) -> None:
        self._probe_in_flight = False
        self._failures += 1
        if self._state is BreakerState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
            self._transition(BreakerState.OPEN)

    def record_ignored(self) -> None:
        """Release a half-open probe whose outcome says nothing about the dependency."""
        self._probe_in_flight = False


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with full jitter: sleep ~ U(0, min(max_delay, base * 2**n))."""

    attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0

    def delay(self, retry_number: int, rng: Callable[[], float] = random.random) -> float:
        return float(rng() * min(self.max_delay, self.base_delay * 2**retry_number))


async def resilient_call[T](
    func: Callable[[], Awaitable[T]],
    *,
    breaker: CircuitBreaker,
    policy: RetryPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `func`, retrying failures listed in `retry_on` through the breaker.

    Every attempt goes through the breaker, so once it opens the remaining retries are
    skipped and `CircuitOpenError` is raised. Other exceptions propagate untouched.
    """
    for attempt in range(policy.attempts):
        breaker.before_call()
        try:
            result = await func()
        except retry_on as exc:
            breaker.record_failure()
            if attempt == policy.attempts - 1:
                raise
            DEPENDENCY_RETRIES.labels(breaker.name).inc()
            logger.warning(
                "dependency_retry",
                name=breaker.name,
                attempt=attempt + 1,
                error=type(exc).__name__,
            )
            await sleep(policy.delay(attempt, rng))
        except BaseException:
            breaker.record_ignored()
            raise
        else:
            breaker.record_success()
            return result
    raise AssertionError("unreachable")  # pragma: no cover


_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, *, failure_threshold: int, recovery_timeout: float) -> CircuitBreaker:
    """Process-wide breaker per dependency, so the readiness probe can report its state."""
    if name not in _BREAKERS:
        _BREAKERS[name] = CircuitBreaker(
            name, failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
        )
    return _BREAKERS[name]


def breaker_states() -> dict[str, str]:
    return {name: b.reported_state.name.lower() for name, b in _BREAKERS.items()}


class TransientStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream returned {status_code}")
        self.status_code = status_code


TRANSIENT_STATUS = frozenset({502, 503, 504})
# A read timeout is not retried: the call already used its whole time budget.
OLLAMA_RETRY_ON: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    TransientStatusError,
)


class ResilientTransport(httpx.AsyncBaseTransport):
    """httpx transport adding retries, backoff and a circuit breaker to every request."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        breaker: CircuitBreaker,
        policy: RetryPolicy,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._inner = inner
        self._breaker = breaker
        self._policy = policy
        self._sleep = sleep
        self._rng = rng

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async def attempt() -> httpx.Response:
            try:
                response = await self._inner.handle_async_request(request)
            except httpx.ReadTimeout:
                # Not retried, but the breaker must still see it.
                self._breaker.record_failure()
                raise
            if response.status_code in TRANSIENT_STATUS:
                await response.aclose()
                raise TransientStatusError(response.status_code)
            return response

        try:
            return await resilient_call(
                attempt,
                breaker=self._breaker,
                policy=self._policy,
                retry_on=OLLAMA_RETRY_ON,
                sleep=self._sleep,
                rng=self._rng,
            )
        except TransientStatusError as exc:
            raise httpx.HTTPStatusError(
                f"{exc.status_code} after {self._policy.attempts} attempts",
                request=request,
                response=httpx.Response(exc.status_code, request=request),
            ) from exc

    async def aclose(self) -> None:
        await self._inner.aclose()
