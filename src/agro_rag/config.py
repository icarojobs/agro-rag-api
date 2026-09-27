from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGRO_", env_file=".env", extra="ignore")

    app_name: str = "agro-rag-api"
    environment: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://agro:agro@localhost:5433/agro"
    db_pool_size: int = 10
    db_max_overflow: int = 10

    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    llm_provider: Literal["ollama", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096

    corpus_dir: Path = Path("corpus")
    collection: str = "default"
    chunk_size: int = 500
    chunk_overlap: int = 80


@lru_cache
def get_settings() -> Settings:
    return Settings()
