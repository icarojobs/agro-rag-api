import os
from collections.abc import AsyncIterator, Iterator

os.environ.setdefault(
    "AGRO_DATABASE_URL", "postgresql+asyncpg://agro:agro@localhost:5433/agro_test"
)

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.db.session import get_engine, get_sessionmaker
from agro_rag.main import create_app


@pytest.fixture(scope="session", autouse=True)
def migrated_database() -> Iterator[None]:
    config = Config("alembic.ini")
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    yield


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
