import time

from fastapi import APIRouter, Response, status

from agro_rag import __version__
from agro_rag.agent import build_agent
from agro_rag.api.deps import CacheDep, EmbeddingsDep, LLMDep, SessionDep, SettingsDep
from agro_rag.api.schemas import (
    AgentResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Source,
)
from agro_rag.cache import make_key
from agro_rag.db.session import get_sessionmaker, ping
from agro_rag.observability import GENERATION_SECONDS
from agro_rag.rag import RagChain
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
