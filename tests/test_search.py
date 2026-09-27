from pathlib import Path

import pytest
from httpx import AsyncClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.db.session import get_sessionmaker
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.ingestion.pipeline import ingest_corpus
from agro_rag.retrieval import PgVectorRetriever, search


@pytest.fixture
async def indexed(session: AsyncSession, corpus_dir: Path) -> None:
    for collection in ("default", "other"):
        await ingest_corpus(
            session,
            HashingEmbeddings(),
            corpus_dir,
            collection=collection,
            chunk_size=500,
            chunk_overlap=50,
        )


@pytest.mark.usefixtures("indexed")
async def test_search_ranks_relevant_document_first(session: AsyncSession) -> None:
    results = await search(
        session, HashingEmbeddings(), "calcário acidez do solo", k=2, collection="default"
    )

    assert [r.source for r in results] == ["solo/calagem.md", "pragas/percevejo.md"]
    assert results[0].score > results[1].score
    assert -1.0 <= results[1].score <= 1.0


@pytest.mark.usefixtures("indexed")
async def test_search_filters_by_category(session: AsyncSession) -> None:
    results = await search(
        session, HashingEmbeddings(), "calcário", k=5, collection="default", category="pragas"
    )

    assert {r.category for r in results} == {"pragas"}


@pytest.mark.usefixtures("indexed")
async def test_search_endpoint(client: AsyncClient) -> None:
    response = await client.post("/search", json={"query": "percevejo na soja", "k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["source"] == "pragas/percevejo.md"
    assert body["took_ms"] >= 0


async def test_search_endpoint_validates_input(client: AsyncClient) -> None:
    response = await client.post("/search", json={"query": "a", "k": 100})

    assert response.status_code == 422


@pytest.mark.usefixtures("indexed")
async def test_langchain_retriever_returns_documents() -> None:
    retriever = PgVectorRetriever(
        sessionmaker=get_sessionmaker(), embeddings=HashingEmbeddings(), collection="other", k=1
    )

    docs = await retriever.ainvoke("calagem e saturação por bases")

    assert docs[0].metadata["source"] == "solo/calagem.md"
    with pytest.raises(NotImplementedError):
        retriever._get_relevant_documents(
            "x", run_manager=CallbackManagerForRetrieverRun.get_noop_manager()
        )
