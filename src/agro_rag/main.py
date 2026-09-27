from fastapi import FastAPI

from agro_rag import __version__
from agro_rag.api.routes import router
from agro_rag.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(router)
    return app


app = create_app()
