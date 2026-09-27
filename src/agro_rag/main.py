from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agro_rag import __version__
from agro_rag.api.routes import router
from agro_rag.config import get_settings
from agro_rag.embeddings import get_embeddings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load the embedding model once at startup instead of on the first request.
    get_embeddings()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
