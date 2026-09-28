import asyncio
from dataclasses import dataclass

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agro_rag.db.models import Chunk, Document
from agro_rag.observability import RETRIEVAL_SECONDS, tracer


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source: str
    title: str
    category: str
    content: str
    score: float

    def to_langchain(self) -> LCDocument:
        return LCDocument(
            page_content=self.content,
            metadata={
                "source": self.source,
                "title": self.title,
                "category": self.category,
                "score": self.score,
            },
        )


async def search_by_vector(
    session: AsyncSession,
    vector: list[float],
    *,
    k: int,
    collection: str,
    category: str | None = None,
) -> list[RetrievedChunk]:
    distance = Chunk.embedding.cosine_distance(vector)
    stmt = (
        select(Document.source, Document.title, Document.category, Chunk.content, distance)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection == collection)
        .order_by(distance)
        .limit(k)
    )
    if category:
        stmt = stmt.where(Document.category == category)

    # Filters are applied after the HNSW scan; iterative scan keeps returning
    # candidates until `k` rows survive the filter.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    rows = (await session.execute(stmt)).all()
    return [
        RetrievedChunk(
            source=source, title=title, category=cat, content=content, score=1.0 - float(dist)
        )
        for source, title, cat, content, dist in rows
    ]


async def search(
    session: AsyncSession,
    embeddings: Embeddings,
    query: str,
    *,
    k: int,
    collection: str,
    category: str | None = None,
) -> list[RetrievedChunk]:
    with (
        RETRIEVAL_SECONDS.time(),
        tracer.start_as_current_span("retrieval.search") as span,
    ):
        span.set_attribute("retrieval.k", k)
        span.set_attribute("retrieval.collection", collection)
        with tracer.start_as_current_span("retrieval.embed_query"):
            vector = await asyncio.to_thread(embeddings.embed_query, query)
        results = await search_by_vector(
            session, vector, k=k, collection=collection, category=category
        )
        span.set_attribute("retrieval.results", len(results))
        if results:
            span.set_attribute("retrieval.top_score", results[0].score)
        return results


class PgVectorRetriever(BaseRetriever):
    """LangChain retriever backed by the pgvector `chunks` table."""

    sessionmaker: async_sessionmaker[AsyncSession]
    embeddings: Embeddings
    collection: str
    k: int = 4

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[LCDocument]:
        async with self.sessionmaker() as session:
            chunks = await search(
                session, self.embeddings, query, k=self.k, collection=self.collection
            )
        return [c.to_langchain() for c in chunks]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[LCDocument]:
        raise NotImplementedError("PgVectorRetriever is async-only; use ainvoke()")
