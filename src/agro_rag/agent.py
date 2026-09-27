import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agro_rag.llm import NO_ANSWER, REWRITE_INSTRUCTION, YES_NO_INSTRUCTION
from agro_rag.rag import format_context
from agro_rag.tools import liming_calculator

TOOLS = [liming_calculator]
MAX_TOOL_CALLS = 3

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você avalia se um documento recuperado é relevante para uma pergunta de "
            "agronomia. Não precisa ser um teste rigoroso: se o documento tratar do mesmo "
            "assunto ou contiver termos da pergunta, ele é relevante. " + YES_NO_INSTRUCTION,
        ),
        ("human", "Pergunta: {question}\nDocumento: {document}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você reformula perguntas de agronomia usando termos técnicos. "
            + REWRITE_INSTRUCTION
            + " Retorne somente a nova pergunta.",
        ),
        ("human", "Pergunta: {question}"),
    ]
)

ANSWER_SYSTEM = """Você é um agrônomo assistente. Responda em português do Brasil, de forma \
direta e em no máximo cinco frases, com base no contexto e cite as fontes pelo número, \
como [1]. Quando a pergunta pedir o cálculo da necessidade de calagem e trouxer V1, V2, CTC \
e PRNT, use a ferramenta liming_calculator. Se o contexto não tiver a resposta, diga que não \
encontrou a informação."""


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    documents: list[Document]
    rewrites: int
    messages: Annotated[list[AnyMessage], add_messages]
    steps: Annotated[list[str], operator.add]
    answer: str


def _yes(text: str) -> bool:
    return text.strip().strip("'\"*.` ").lower().startswith(("sim", "yes"))


def build_agent(
    retriever: BaseRetriever, llm: BaseChatModel, max_rewrites: int = 1
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """retrieve -> grade -> (rewrite -> retrieve)* -> answer <-> tools."""
    grader = GRADE_PROMPT | llm | StrOutputParser()
    rewriter = REWRITE_PROMPT | llm | StrOutputParser()
    answerer = llm.bind_tools(TOOLS)

    async def retrieve(state: AgentState) -> AgentState:
        query = state.get("search_query") or state["question"]
        docs = await retriever.ainvoke(query)
        return {"documents": docs, "steps": [f"retrieve:{len(docs)}"]}

    async def grade(state: AgentState) -> AgentState:
        docs = state["documents"]
        verdicts = await grader.abatch(
            [{"question": state["question"], "document": d.page_content} for d in docs]
        )
        relevant = [d for d, v in zip(docs, verdicts, strict=True) if _yes(v)]
        return {"documents": relevant, "steps": [f"grade:{len(relevant)}"]}

    def route_after_grade(state: AgentState) -> Literal["answer", "rewrite"]:
        if state["documents"] or state.get("rewrites", 0) >= max_rewrites:
            return "answer"
        return "rewrite"

    async def rewrite(state: AgentState) -> AgentState:
        query = await rewriter.ainvoke({"question": state["question"]})
        return {
            "search_query": query.strip() or state["question"],
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": ["rewrite"],
        }

    async def answer(state: AgentState) -> AgentState:
        messages: list[AnyMessage] = state.get("messages") or [
            SystemMessage(ANSWER_SYSTEM),
            HumanMessage(
                f"Contexto:\n{format_context(state['documents'])}\n\nPergunta: {state['question']}"
            ),
        ]
        reply = await answerer.ainvoke(messages)
        new_messages: list[AnyMessage] = [reply] if state.get("messages") else [*messages, reply]
        return {"messages": new_messages, "steps": ["answer"]}

    def route_after_answer(state: AgentState) -> Literal["tools", "finish"]:
        last = state["messages"][-1]
        tool_rounds = sum(1 for s in state.get("steps", []) if s == "tools")
        if isinstance(last, AIMessage) and last.tool_calls and tool_rounds < MAX_TOOL_CALLS:
            return "tools"
        return "finish"

    async def finish(state: AgentState) -> AgentState:
        content = state["messages"][-1].content
        text = content if isinstance(content, str) else str(content)
        return {"answer": text.strip() or NO_ANSWER}

    tool_node = ToolNode(TOOLS)

    async def tools(state: AgentState) -> AgentState:
        result = await tool_node.ainvoke(state)
        return {"messages": result["messages"], "steps": ["tools"]}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)
    graph.add_node("tools", tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("answer", route_after_answer)
    graph.add_edge("tools", "answer")
    graph.add_edge("finish", END)
    return graph.compile()
