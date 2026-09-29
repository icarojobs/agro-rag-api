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
    redis_retry_attempts: int = 2
    redis_retry_base_delay_seconds: float = 0.02
    redis_breaker_failures: int = 3
    redis_breaker_recovery_seconds: float = 10.0
    cache_ttl_seconds: int = 300
    cache_embedding_ttl_seconds: int = 3600

    # AWS messaging. `aws_endpoint_url` points at the local emulator (floci); leave it unset
    # to use the real AWS credential chain. Ingestion via SQS is disabled without a URL or
    # `aws_enabled`.
    aws_endpoint_url: str | None = None
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    ingestion_topic: str = "agro-ingestion"
    ingestion_queue: str = "agro-ingestion-jobs"
    ingestion_dlq: str = "agro-ingestion-dlq"
    ingestion_audit_queue: str = "agro-ingestion-audit"
    ingestion_table: str = "agro-ingestion-jobs"
    ingestion_visibility_timeout_seconds: int = 60
    ingestion_max_receive_count: int = 3
    ingestion_poll_wait_seconds: int = 10
    ingestion_batch_size: int = 5

    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"

    llm_provider: Literal["ollama", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096
    ollama_connect_timeout_seconds: float = 3.0
    ollama_timeout_seconds: float = 120.0
    ollama_retry_attempts: int = 3
    ollama_retry_base_delay_seconds: float = 0.3
    ollama_breaker_failures: int = 5
    ollama_breaker_recovery_seconds: float = 30.0

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "agro-rag-retrieval"

    corpus_dir: Path = Path("corpus")
    collection: str = "default"
    chunk_size: int = 500
    chunk_overlap: int = 80


@lru_cache
def get_settings() -> Settings:
    return Settings()
