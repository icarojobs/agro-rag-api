import time
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Response, status

from agro_rag import __version__
from agro_rag.agent import build_agent
from agro_rag.api.deps import (
    CacheDep,
    EmbeddingsDep,
    IngestionQueueDep,
    LLMDep,
    SessionDep,
    SettingsDep,
)
from agro_rag.api.schemas import (
    AgentResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestAccepted,
    IngestJob,
    IngestRequest,
    LivenessResponse,
    ReadinessResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Source,
)
from agro_rag.cache import RedisCache, make_key
from agro_rag.db.session import get_sessionmaker, ping
from agro_rag.ingestion.jobs import IngestionMessage
from agro_rag.observability import GENERATION_SECONDS
from agro_rag.rag import RagChain
from agro_rag.resilience import breaker_states
from agro_rag.retrieval import PgVectorRetriever, search

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


async def _cache_status(cache: RedisCache | None) -> str:
    if cache is None:
        return "disabled"
    return "ok" if await cache.ping() else "unavailable"


@router.get("/livez", response_model=LivenessResponse, tags=["ops"])
async def liveness() -> LivenessResponse:
    """The process is up. It checks no dependency, so a slow database never restarts the pod."""
    return LivenessResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse, tags=["ops"])
async def readiness(
    session: SessionDep, response: Response, settings: SettingsDep, cache: CacheDep
) -> ReadinessResponse:
    """Ready when the database answers. Redis and Ollama only degrade the service, so they are
    reported but do not take the pod out of rotation."""
    db_ok = await ping(session)
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    cache_status = await _cache_status(cache)
    if settings.llm_provider == "fake":
        llm_status = "fake"
    else:
        llm_status = "unavailable" if breaker_states().get("ollama") == "open" else "ok"
    return ReadinessResponse(
        status="ready" if db_ok else "not_ready",
        version=__version__,
        database="ok" if db_ok else "unavailable",
        cache=cache_status,
        llm=llm_status,
    )


@router.post("/search", response_model=SearchResponse, tags=["retrieval"])
async def semantic_search(
    body: SearchRequest,
    session: SessionDep,
    embeddings: EmbeddingsDep,
    settings: SettingsDep,
    cache: CacheDep,
) -> SearchResponse:
    started = time.perf_counter()
    chunks = await search(
        session,
        embeddings,
        body.query,
        k=body.k,
        collection=body.collection or settings.collection,
        category=body.category,
        cache=cache,
    )
    return SearchResponse(
        query=body.query,
        results=[SearchHit.model_validate(c, from_attributes=True) for c in chunks],
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.post("/ask", response_model=AskResponse, tags=["generation"])
async def ask(
    body: AskRequest,
    embeddings: EmbeddingsDep,
    llm: LLMDep,
    settings: SettingsDep,
    cache: CacheDep,
) -> AskResponse:
    started = time.perf_counter()
    model = getattr(llm, "model", llm._llm_type)
    key = make_key(
        "answer", model, settings.collection, body.k, body.question, scope=settings.collection
    )
    if cache is not None and (cached := await cache.get("answer", key)) is not None:
        return AskResponse(
            answer=cached["answer"],
            sources=[Source(**s) for s in cached["sources"]],
            model=model,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    retriever = PgVectorRetriever(
        sessionmaker=get_sessionmaker(),
        embeddings=embeddings,
        collection=settings.collection,
        k=body.k,
        cache=cache,
    )
    with GENERATION_SECONDS.labels("ask").time():
        result = await RagChain(retriever, llm).ainvoke(body.question)
    sources = [
        Source(source=d.metadata["source"], title=d.metadata["title"], score=d.metadata["score"])
        for d in result.documents
    ]
    if cache is not None:
        await cache.set(
            "answer",
            key,
            {"answer": result.answer, "sources": [s.model_dump() for s in sources]},
            settings.cache_ttl_seconds,
        )
    return AskResponse(
        answer=result.answer,
        sources=sources,
        model=model,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.post("/agent", response_model=AgentResponse, tags=["generation"])
async def agent(
    body: AskRequest,
    embeddings: EmbeddingsDep,
    llm: LLMDep,
    settings: SettingsDep,
    cache: CacheDep,
) -> AgentResponse:
    started = time.perf_counter()
    retriever = PgVectorRetriever(
        sessionmaker=get_sessionmaker(),
        embeddings=embeddings,
        collection=settings.collection,
        k=body.k,
        cache=cache,
    )
    with GENERATION_SECONDS.labels("agent").time():
        state = await build_agent(retriever, llm).ainvoke({"question": body.question})
    return AgentResponse(
        answer=state["answer"],
        sources=[
            Source(
                source=d.metadata["source"], title=d.metadata["title"], score=d.metadata["score"]
            )
            for d in state["documents"]
        ],
        steps=state["steps"],
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )


_QUEUE_OFF = HTTPException(
    status.HTTP_503_SERVICE_UNAVAILABLE, "Asynchronous ingestion is not configured."
)


@router.post(
    "/ingest/documents",
    response_model=IngestAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
)
async def enqueue_document(
    body: IngestRequest,
    queue: IngestionQueueDep,
    settings: SettingsDep,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
) -> IngestAccepted:
    """Queue a document for indexing. A repeated `Idempotency-Key` returns the same job."""
    if queue is None:
        raise _QUEUE_OFF
    fields = body.model_dump(exclude={"collection"})
    message = IngestionMessage(
        job_id="", collection=body.collection or settings.collection, **fields
    )
    job_id, _ = await queue.enqueue(message, idempotency_key=idempotency_key)
    return IngestAccepted(job_id=job_id, status="queued")


@router.get("/ingest/jobs/{job_id}", response_model=IngestJob, tags=["ingestion"])
async def ingestion_job(job_id: str, queue: IngestionQueueDep) -> IngestJob:
    if queue is None:
        raise _QUEUE_OFF
    job = await queue.job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")
    return IngestJob.model_validate(job)
