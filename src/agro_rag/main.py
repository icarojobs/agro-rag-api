from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from agro_rag import __version__
from agro_rag.api.routes import router
from agro_rag.config import get_settings
from agro_rag.db.session import get_engine
from agro_rag.embeddings import get_embeddings
from agro_rag.llm import LLM_UNAVAILABLE_ERRORS
from agro_rag.observability import (
    configure_logging,
    configure_metrics,
    configure_tracing,
    request_logging_middleware,
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load the embedding model once at startup instead of on the first request.
    get_embeddings()
    structlog.get_logger("agro_rag").info("startup_complete", version=__version__)
    yield
    await get_engine().dispose()


async def llm_unavailable(_: Request, exc: Exception) -> JSONResponse:
    structlog.get_logger("agro_rag").warning("llm_unavailable", error=type(exc).__name__)
    return JSONResponse(
        {"detail": "The language model is unavailable, try again shortly."},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": "30"},
    )


async def queue_unavailable(_: Request, exc: Exception) -> JSONResponse:
    structlog.get_logger("agro_rag").warning("queue_unavailable", error=type(exc).__name__)
    return JSONResponse(
        {"detail": "The ingestion queue is unavailable, try again shortly."},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": "10"},
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.middleware("http")(request_logging_middleware)
    app.include_router(router)
    for aws_error in (BotoCoreError, ClientError):
        app.add_exception_handler(aws_error, queue_unavailable)
    for error in LLM_UNAVAILABLE_ERRORS:
        app.add_exception_handler(error, llm_unavailable)
    configure_metrics(app)
    configure_tracing(app, get_engine(), settings)
    return app


app = create_app()
