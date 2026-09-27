from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag import __version__
from agro_rag.api.schemas import HealthResponse
from agro_rag.db.session import get_session, ping

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(session: SessionDep, response: Response) -> HealthResponse:
    db_ok = await ping(session)
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if db_ok else "degraded",
        version=__version__,
        database="ok" if db_ok else "unavailable",
    )
