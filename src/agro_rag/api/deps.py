from typing import Annotated

from fastapi import Depends
from langchain_core.embeddings import Embeddings
from sqlalchemy.ext.asyncio import AsyncSession

from agro_rag.config import Settings, get_settings
from agro_rag.db.session import get_session
from agro_rag.embeddings import get_embeddings

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingsDep = Annotated[Embeddings, Depends(get_embeddings)]
