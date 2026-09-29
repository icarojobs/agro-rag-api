from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGRO_", env_file=".env", extra="ignore")

    app_name: str = "agro-rag-api"
    environment: str = "local"
    log_level: str = "INFO"
    log_json: bool = True

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None

    database_url: str = "postgresql+asyncpg://agro:agro@localhost:5433/agro"
    db_pool_size: int = 10
    db_max_overflow: int = 10

    # Cache is optional: leave unset to run without Redis.
    redis_url: str | None = None
    redis_timeout_seconds: float = 0.25
    cache_ttl_seconds: int = 300
    cache_embedding_ttl_seconds: int = 3600

    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"

    llm_provider: Literal["ollama", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "agro-rag-retrieval"

    corpus_dir: Path = Path("corpus")
    collection: str = "default"
    chunk_size: int = 500
    chunk_overlap: int = 80


@lru_cache
def get_settings() -> Settings:
    return Settings()
