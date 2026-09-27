from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGRO_", env_file=".env", extra="ignore")

    app_name: str = "agro-rag-api"
    environment: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://agro:agro@localhost:5433/agro"
    db_pool_size: int = 10
    db_max_overflow: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
