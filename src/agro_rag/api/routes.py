import time

from fastapi import APIRouter, Response, status

from agro_rag import __version__
from agro_rag.api.deps import EmbeddingsDep, SessionDep, SettingsDep
from agro_rag.api.schemas import HealthResponse, SearchHit, SearchRequest, SearchResponse
from agro_rag.db.session import ping
from agro_rag.retrieval import search

router = APIRouter()


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


@router.post("/search", response_model=SearchResponse, tags=["retrieval"])
async def semantic_search(
    body: SearchRequest, session: SessionDep, embeddings: EmbeddingsDep, settings: SettingsDep
) -> SearchResponse:
    started = time.perf_counter()
    chunks = await search(
        session,
        embeddings,
        body.query,
        k=body.k,
        collection=body.collection or settings.collection,
        category=body.category,
    )
    return SearchResponse(
        query=body.query,
        results=[SearchHit.model_validate(c, from_attributes=True) for c in chunks],
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )
