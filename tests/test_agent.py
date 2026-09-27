from pathlib import Path

import pytest
from httpx import AsyncClient
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.agent import build_agent
from agro_rag.embeddings import HashingEmbeddings
from agro_rag.ingestion.pipeline import ingest_corpus
from agro_rag.llm import NO_ANSWER, DeterministicChatModel
from agro_rag.tools import liming_calculator


class StaticRetriever(BaseRetriever):
    docs: list[Document]
    queries: list[str] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: object) -> list[Document]:
        raise NotImplementedError

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return self.docs


CALAGEM = Document(
    page_content="Calagem\nA calagem corrige a acidez do solo com calcário. Outra frase.",
    metadata={"title": "Calagem", "source": "solo/calagem.md", "score": 0.9},
)
PRAGA = Document(
    page_content="Percevejo\nO percevejo suga grãos de soja.",
    metadata={"title": "Percevejo", "source": "pragas/percevejo.md", "score": 0.4},
)


@pytest.mark.parametrize(
    ("v1", "v2", "ctc", "prnt", "expected"),
    [
        (35, 60, 8, 80, "NC = (60 - 35) x 8 / 80 = 2.50 t/ha de calcário."),
        (70, 60, 8, 80, "não há necessidade de calagem"),
        (120, 60, 8, 80, "Erro: V1 e V2"),
        (35, 60, 0, 80, "Erro: CTC"),
    ],
)
def test_liming_calculator(v1: float, v2: float, ctc: float, prnt: float, expected: str) -> None:
    result = liming_calculator.invoke({"v1": v1, "v2": v2, "ctc": ctc, "prnt": prnt})

    assert expected in result


async def test_agent_filters_irrelevant_documents_and_answers() -> None:
    retriever = StaticRetriever(docs=[PRAGA, CALAGEM])
    agent = build_agent(retriever, DeterministicChatModel())

    state = await agent.ainvoke({"question": "como a calagem corrige a acidez do solo?"})

    assert state["steps"] == ["retrieve:2", "grade:1", "answer"]
    assert [d.metadata["source"] for d in state["documents"]] == ["solo/calagem.md"]
    assert state["answer"] == "A calagem corrige a acidez do solo com calcário. [1]"


async def test_agent_rewrites_once_then_gives_up() -> None:
    retriever = StaticRetriever(docs=[PRAGA], queries=[])
    agent = build_agent(retriever, DeterministicChatModel(), max_rewrites=1)

    state = await agent.ainvoke({"question": "qual a janela de semeadura do trigo?"})

    assert state["steps"] == [
        "retrieve:1",
        "grade:0",
        "rewrite",
        "retrieve:1",
        "grade:0",
        "answer",
    ]
    assert len(retriever.queries) == 2
    assert state["answer"] == NO_ANSWER


async def test_agent_calls_liming_tool() -> None:
    agent = build_agent(StaticRetriever(docs=[CALAGEM]), DeterministicChatModel())

    state = await agent.ainvoke(
        {"question": "Calcule a calagem do solo com V1 de 35%, V2 de 60%, CTC de 8 e PRNT de 80%"}
    )

    assert state["steps"][-3:] == ["answer", "tools", "answer"]
    assert state["answer"] == "NC = (60 - 35) x 8 / 80 = 2.50 t/ha de calcário."


async def test_agent_stops_after_max_tool_rounds() -> None:
    class LoopingModel(DeterministicChatModel):
        def _respond(self, messages: list) -> AIMessage:  # type: ignore[type-arg]
            if "Pergunta:" in str(messages[-1].content) and "Documento:" in str(
                messages[-1].content
            ):
                return AIMessage(content="sim")
            args = {"v1": 30, "v2": 60, "ctc": 10, "prnt": 100}
            return AIMessage(
                content="ainda calculando",
                tool_calls=[{"name": "liming_calculator", "args": args, "id": "c"}],
            )

    llm: BaseChatModel = LoopingModel()
    state = await build_agent(StaticRetriever(docs=[CALAGEM]), llm).ainvoke(
        {"question": "calagem do solo"}
    )

    assert state["steps"].count("tools") == 3
    assert state["answer"] == "ainda calculando"


async def test_agent_endpoint(client: AsyncClient, session: AsyncSession, corpus_dir: Path) -> None:
    await ingest_corpus(
        session,
        HashingEmbeddings(),
        corpus_dir,
        collection="default",
        chunk_size=500,
        chunk_overlap=50,
    )

    response = await client.post(
        "/agent", json={"question": "como o calcário corrige a acidez do solo?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["source"] == "solo/calagem.md"
    assert body["steps"][0] == "retrieve:2"
    assert body["answer"].endswith("[1]")
