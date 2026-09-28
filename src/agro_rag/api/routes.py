import time

from fastapi import APIRouter, Response, status

from agro_rag import __version__
from agro_rag.agent import build_agent
from agro_rag.api.deps import EmbeddingsDep, LLMDep, SessionDep, SettingsDep
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


@router.post("/ask", response_model=AskResponse, tags=["generation"])
async def ask(
    body: AskRequest, embeddings: EmbeddingsDep, llm: LLMDep, settings: SettingsDep
) -> AskResponse:
    started = time.perf_counter()
    retriever = PgVectorRetriever(
        sessionmaker=get_sessionmaker(),
        embeddings=embeddings,
        collection=settings.collection,
        k=body.k,
    )
    with GENERATION_SECONDS.labels("ask").time():
        result = await RagChain(retriever, llm).ainvoke(body.question)
    return AskResponse(
        answer=result.answer,
        sources=[
            Source(
                source=d.metadata["source"], title=d.metadata["title"], score=d.metadata["score"]
            )
            for d in result.documents
        ],
        model=getattr(llm, "model", llm._llm_type),
        took_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.post("/agent", response_model=AgentResponse, tags=["generation"])
async def agent(
    body: AskRequest, embeddings: EmbeddingsDep, llm: LLMDep, settings: SettingsDep
) -> AgentResponse:
    started = time.perf_counter()
    retriever = PgVectorRetriever(
        sessionmaker=get_sessionmaker(),
        embeddings=embeddings,
        collection=settings.collection,
        k=body.k,
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
