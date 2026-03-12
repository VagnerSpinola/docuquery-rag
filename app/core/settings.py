from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="Neural Knowledge Engine", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    api_prefix: str = Field(default="", alias="API_PREFIX")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json_format: bool = Field(default=True, alias="LOG_JSON_FORMAT")
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")

    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    openai_temperature: float = Field(default=0.0, alias="OPENAI_TEMPERATURE")
    openai_request_timeout: int = Field(default=60, alias="OPENAI_REQUEST_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_retry_backoff_seconds: float = Field(default=1.0, alias="LLM_RETRY_BACKOFF_SECONDS")

    documents_directory: Path = Field(default=ROOT_DIR / "data" / "documents", alias="DOCUMENTS_DIRECTORY")
    chroma_persist_directory: Path = Field(default=ROOT_DIR / "data" / "chroma", alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="neural_knowledge_documents", alias="CHROMA_COLLECTION_NAME")
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")
    chunk_size_tokens: int = Field(default=800, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=120, alias="CHUNK_OVERLAP_TOKENS")
    context_max_documents: int = Field(default=6, alias="CONTEXT_MAX_DOCUMENTS")
    context_max_characters: int = Field(default=6000, alias="CONTEXT_MAX_CHARACTERS")
    hybrid_vector_weight: float = Field(default=0.65, alias="HYBRID_VECTOR_WEIGHT")
    hybrid_keyword_weight: float = Field(default=0.35, alias="HYBRID_KEYWORD_WEIGHT")

    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    database_url: str = Field(
        default="postgresql+psycopg://neural:neural@postgres:5432/neural_knowledge_engine",
        alias="DATABASE_URL",
    )

    celery_broker_url: str = Field(default="redis://redis:6379/1", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://redis:6379/2", alias="CELERY_RESULT_BACKEND")

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.documents_directory.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    return settings