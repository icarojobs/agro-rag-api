from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

SYSTEM_PROMPT = """Você é um assistente técnico de agronomia. Responda em português do Brasil, \
de forma direta e em no máximo cinco frases, usando somente as informações do contexto. Cite \
as fontes pelo número, como [1] ou [2]. Se o contexto não tiver a resposta, diga que não \
encontrou a informação na base de conhecimento."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Contexto:\n{context}\n\nPergunta: {question}"),
    ]
)


def format_context(docs: list[Document]) -> str:
    return "\n".join(
        f"[{i}] {d.metadata.get('title', '')} ({d.metadata.get('source', '')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer: str
    documents: list[Document]


class RagChain:
    """Retrieve -> stuff the context into the prompt -> generate."""

    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel) -> None:
        self.retriever = retriever
        self.generate = PROMPT | llm | StrOutputParser()

    async def ainvoke(self, question: str) -> RagAnswer:
        docs = await self.retriever.ainvoke(question)
        answer = await self.generate.ainvoke(
            {"context": format_context(docs), "question": question}
        )
        return RagAnswer(answer=answer.strip(), documents=docs)
