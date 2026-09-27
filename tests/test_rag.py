from pathlib import Path

import pytest
from httpx import AsyncClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.config import get_settings
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.ingestion.pipeline import ingest_corpus
from agro_rag.llm import NO_ANSWER, DeterministicChatModel, get_llm
from agro_rag.rag import format_context


def _doc(content: str, source: str) -> Document:
    return Document(page_content=content, metadata={"title": "T", "source": source})


def test_format_context_numbers_passages() -> None:
    context = format_context([_doc("a", "x.md"), _doc("b", "y.md")])

    assert context == "[1] T (x.md)\na\n[2] T (y.md)\nb"


async def test_fake_llm_quotes_first_passage() -> None:
    context = format_context([_doc("Calagem\nO calcário corrige a acidez. Outra frase.", "x.md")])

    reply = await DeterministicChatModel().ainvoke([HumanMessage(f"Contexto:\n{context}")])

    assert reply.content == "O calcário corrige a acidez. [1]"


async def test_fake_llm_without_context_declines() -> None:
    llm = DeterministicChatModel()

    reply = await llm.bind_tools([]).ainvoke([HumanMessage("Pergunta sem contexto")])

    assert reply.content == NO_ANSWER


def test_get_llm_builds_ollama_client(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(get_llm(), DeterministicChatModel)

    get_llm.cache_clear()
    get_settings.cache_clear()
    monkeypatch.setenv("AGRO_LLM_PROVIDER", "ollama")
    llm = get_llm()

    assert isinstance(llm, ChatOllama)
    assert llm.model == get_settings().ollama_model


async def test_ask_endpoint_answers_with_sources(
    client: AsyncClient, session: AsyncSession, corpus_dir: Path
) -> None:
    await ingest_corpus(
        session,
        HashingEmbeddings(),
        corpus_dir,
        collection="default",
        chunk_size=500,
        chunk_overlap=50,
    )

    response = await client.post("/ask", json={"question": "o que o calcário corrige?", "k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "O calcário corrige a acidez do solo e fornece cálcio e magnésio. [1]"
    assert body["sources"][0]["source"] == "solo/calagem.md"
    assert body["model"] == "deterministic-fake"
