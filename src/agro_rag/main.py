from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from agro_rag import __version__
from agro_rag.api.routes import router
from agro_rag.config import get_settings
from agro_rag.db.session import get_engine
from agro_rag.embeddings import get_embeddings
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


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.middleware("http")(request_logging_middleware)
    app.include_router(router)
    configure_metrics(app)
    configure_tracing(app, get_engine(), settings)
    return app


app = create_app()
