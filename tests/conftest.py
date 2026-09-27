from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from agro_rag.main import create_app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
