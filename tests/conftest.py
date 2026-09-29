import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

os.environ.setdefault(
    "AGRO_DATABASE_URL", "postgresql+asyncpg://agro:agro@localhost:5433/agro_test"
)
os.environ.setdefault("AGRO_EMBEDDING_PROVIDER", "hashing")
os.environ.setdefault("AGRO_LLM_PROVIDER", "fake")

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.cache import get_cache
from agro_rag.config import get_settings
from agro_rag.db.session import get_engine, get_sessionmaker
from agro_rag.embeddings import get_embeddings
from agro_rag.ingestion.queue import get_ingestion_queue
from agro_rag.llm import get_llm
from agro_rag.main import create_app


@pytest.fixture(scope="session", autouse=True)
def migrated_database() -> Iterator[None]:
    config = Config("alembic.ini")
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    yield


@pytest.fixture(autouse=True)
def reset_cached_settings() -> Iterator[None]:
    yield
    get_settings.cache_clear()
    get_cache.cache_clear()
    get_ingestion_queue.cache_clear()
    get_embeddings.cache_clear()
    get_llm.cache_clear()


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as s:
        yield s
        await s.rollback()


@pytest.fixture(autouse=True)
async def clean_tables() -> AsyncIterator[None]:
    yield
    async with get_engine().begin() as conn:
        await conn.execute(text("TRUNCATE documents, chunks CASCADE"))


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "solo").mkdir()
    (tmp_path / "pragas").mkdir()
    (tmp_path / "solo" / "calagem.md").write_text(
        "# Calagem\n\nO calcário corrige a acidez do solo e fornece cálcio e magnésio.\n\n"
        "A dose é calculada pelo método da saturação por bases.",
        encoding="utf-8",
    )
    (tmp_path / "pragas" / "percevejo.md").write_text(
        "# Percevejo-marrom\n\nO percevejo suga os grãos de soja entre R3 e R6.",
        encoding="utf-8",
    )
    return tmp_path
