from typing import Annotated

from fastapi import Depends
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.config import Settings, get_settings
from agro_rag.db.session import get_session
from agro_rag.embeddings import get_embeddings
from agro_rag.llm import get_llm

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingsDep = Annotated[Embeddings, Depends(get_embeddings)]
LLMDep = Annotated[BaseChatModel, Depends(get_llm)]
