import json
import logging

import pytest
import structlog
from fastapi import FastAPI
from httpx import AsyncClient
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agro_rag import observability
from agro_rag.config import Settings
from agro_rag.db.session import get_engine
from agro_rag.observability import configure_logging, configure_tracing


async def test_metrics_endpoint_exposes_custom_metrics(client: AsyncClient) -> None:
    await client.post("/search", json={"query": "adubação potássica", "k": 2})

    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "agro_rag_retrieval_seconds_count" in response.text
    assert "http_request_duration_seconds" in response.text


async def test_request_id_is_propagated(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"x-request-id": "abc123"})

    assert response.headers["x-request-id"] == "abc123"


def test_json_logging(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(log_json=True))

    structlog.get_logger("test").info("hello", crop="soja")
    logging.getLogger("stdlib").warning("from stdlib")

    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["event"] == "hello"
    assert lines[0]["crop"] == "soja"
    assert lines[1]["level"] == "warning"


def test_console_logging(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(log_json=False))

    structlog.get_logger("test").info("hello")

    assert "hello" in capsys.readouterr().out


def test_tracing_is_opt_in() -> None:
    app = FastAPI()
    configure_tracing(app, get_engine(), Settings(otel_enabled=False))

    assert not getattr(app, "_is_instrumented_by_opentelemetry", False)


def test_tracing_instruments_app_and_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoints: list[str] = []

    def fake_exporter(endpoint: str) -> InMemorySpanExporter:
        endpoints.append(endpoint)
        return InMemorySpanExporter()

    monkeypatch.setattr(observability, "OTLPSpanExporter", fake_exporter)
    app = FastAPI()
    settings = Settings(otel_enabled=True, otel_exporter_otlp_endpoint="http://jaeger:4318")
    try:
        configure_tracing(app, get_engine(), settings)
        assert app._is_instrumented_by_opentelemetry  # type: ignore[attr-defined]
        assert endpoints == ["http://jaeger:4318/v1/traces"]
    finally:
        SQLAlchemyInstrumentor().uninstrument()
