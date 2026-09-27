from fastapi import APIRouter

from agro_rag import __version__
from agro_rag.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
