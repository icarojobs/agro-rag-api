import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import Field

from agro_rag.config import get_settings

NO_ANSWER = "Não encontrei informações suficientes na base de conhecimento para responder."
YES_NO_INSTRUCTION = "Responda apenas 'sim' ou 'não'."
REWRITE_INSTRUCTION = "Reescreva a pergunta para melhorar a busca."

_PASSAGE_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.+?)(?=\n\[\d+\]|\Z)", re.MULTILINE | re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w{4,}", re.UNICODE)
_LIMING_ARGS = ("v1", "v2", "ctc", "prnt")


def _number_after(label: str, text: str) -> float | None:
    match = re.search(rf"\b{label}\b\D{{0,15}}?(\d+(?:[.,]\d+)?)", text, re.IGNORECASE)
    return float(match.group(1).replace(",", ".")) if match else None


def _field(text: str, label: str, *, to_end: bool = False) -> str:
    pattern = rf"^{label}:\s*(.+\Z)" if to_end else rf"^{label}:\s*(.+)$"
    flags = re.MULTILINE | (re.DOTALL if to_end else 0)
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else ""


def _first_sentence(text: str) -> str:
    body = text.strip().split("\n", 1)[-1].strip()
    return _SENTENCE_RE.split(body, maxsplit=1)[0]


class DeterministicChatModel(BaseChatModel):
    """Offline chat model for tests and CI.

    It answers by quoting the first sentence of the first passage found in the
    prompt, so the output is fully determined by the retrieved context. Grading
    uses word overlap and tool calls are emitted only for the liming calculator.
    """

    tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-fake"

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        prompt = "\n".join(str(m.content) for m in messages)
        if YES_NO_INSTRUCTION in prompt:
            question = set(_WORD_RE.findall(_field(prompt, "Pergunta").lower()))
            document = set(_WORD_RE.findall(_field(prompt, "Documento", to_end=True).lower()))
            return AIMessage(content="sim" if len(question & document) >= 2 else "não")
        if REWRITE_INSTRUCTION in prompt:
            return AIMessage(content=_field(prompt, "Pergunta"))

        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_results:
            return AIMessage(content=str(tool_results[-1].content))
        if "liming_calculator" in self.tool_names:
            asked = _field(prompt, "Pergunta")
            args = {name: _number_after(name, asked) for name in _LIMING_ARGS}
            if all(v is not None for v in args.values()):
                call = ToolCall(name="liming_calculator", args=args, id="call_liming")
                return AIMessage(content="", tool_calls=[call])

        passages = _PASSAGE_RE.findall(prompt)
        if not passages:
            return AIMessage(content=NO_ANSWER)
        number, passage = passages[0]
        return AIMessage(content=f"{_first_sentence(passage)} [{number}]")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        names = [t.name if isinstance(t, BaseTool) else getattr(t, "__name__", "") for t in tools]
        return self.model_copy(update={"tool_names": names})


@lru_cache
def get_llm() -> BaseChatModel:
    settings = get_settings()
    if settings.llm_provider == "fake":
        return DeterministicChatModel()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
    )
