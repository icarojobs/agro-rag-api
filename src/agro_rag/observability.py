import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.ext.asyncio import AsyncEngine

from agro_rag import __version__
from agro_rag.config import Settings

RETRIEVAL_SECONDS = Histogram(
    "agro_rag_retrieval_seconds",
    "Time spent embedding the query and searching pgvector",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
GENERATION_SECONDS = Histogram(
    "agro_rag_generation_seconds",
    "End-to-end latency of LLM-backed endpoints",
    ["endpoint"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 20, 40, 80),
)
TOOL_CALLS = Counter("agro_rag_tool_calls_total", "Tool calls executed by the agent", ["tool"])

tracer = trace.get_tracer("agro_rag")


def _add_trace_ids(
    _: object, __: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def configure_logging(settings: Settings) -> None:
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_trace_ids,
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[*shared, structlog.stdlib.ExtraAdder()],
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    # Replaced by the request logging middleware below.
    logging.getLogger("uvicorn.access").disabled = True


def configure_tracing(app: FastAPI, engine: AsyncEngine, settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: settings.app_name, SERVICE_VERSION: __version__})
    )
    if settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls="health,metrics"
    )
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine, tracer_provider=provider)


def configure_metrics(app: FastAPI) -> None:
    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )


async def request_logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()
    log = structlog.get_logger("agro_rag.http")
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", method=request.method, path=request.url.path)
        raise
    finally:
        structlog.contextvars.unbind_contextvars("request_id")
    response.headers["x-request-id"] = request_id
    if request.url.path not in ("/health", "/metrics"):
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    return response
