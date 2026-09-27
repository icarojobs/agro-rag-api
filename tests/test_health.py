from collections.abc import AsyncIterator

from httpx import ASGITransport, AsyncClient

from agro_rag.db.session import get_session
from agro_rag.main import create_app


class _BrokenSession:
    async def execute(self, *_: object) -> None:
        raise ConnectionError("db down")


async def _broken_session() -> AsyncIterator[_BrokenSession]:
    yield _BrokenSession()


async def test_health_reports_database_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0", "database": "ok"}


async def test_health_degrades_when_database_is_down() -> None:
    app = create_app()
    app.dependency_overrides[get_session] = _broken_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/health")

    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"
