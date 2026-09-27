import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from agro_rag.config import get_settings

NO_ANSWER = "Não encontrei informações suficientes na base de conhecimento para responder."

_PASSAGE_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.+?)(?=\n\[\d+\]|\Z)", re.MULTILINE | re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str) -> str:
    body = text.strip().split("\n", 1)[-1].strip()
    return _SENTENCE_RE.split(body, maxsplit=1)[0]


class DeterministicChatModel(BaseChatModel):
    """Offline chat model for tests and CI.

    It answers by quoting the first sentence of the first passage found in the
    prompt, so the output is fully determined by the retrieved context.
    """

    @property
    def _llm_type(self) -> str:
        return "deterministic-fake"

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        prompt = "\n".join(str(m.content) for m in messages)
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
        return self


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
