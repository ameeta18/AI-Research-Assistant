# src/config.py
from functools import lru_cache
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    google_api_key: SecretStr | None = None
    semantic_scholar_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    redis_url: SecretStr | None = None
    database_url: SecretStr | None = None

    langsmith_tracing: bool = False
    langsmith_project: str = "ai-research-assistant"
    langsmith_tracing_sampling_rate: float = 1.0
    langsmith_hide_inputs: bool = True
    langsmith_hide_outputs: bool = True

    cache_enabled: bool = False
    cache_ttl_seconds: int = 900
    cache_key_prefix: str = "ai-researcher"
    redis_connect_timeout_seconds: float = 1.0
    redis_socket_timeout_seconds: float = 1.0

    database_enabled: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 5
    database_pool_timeout_seconds: float = 10.0
    database_connect_timeout_seconds: float = 5.0

    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    embedding_model: str = "models/gemini-embedding-001"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    arxiv_max_results: int = 5
    session_ttl_minutes: int = 60
    max_active_sessions: int = 100
    http_connect_timeout_seconds: float = 5.0
    http_read_timeout_seconds: float = 30.0
    http_max_attempts: int = 3
    http_backoff_seconds: float = 1.0
    http_jitter_seconds: float = 0.5
    http_max_retry_delay_seconds: float = 30.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    max_search_query_chars: int = 500
    max_search_results: int = 10
    max_chat_message_chars: int = 10_000
    max_paper_title_chars: int = 500
    max_retrieval_k: int = 10
    max_pdf_bytes: int = 20_000_000
    max_pdf_pages: int = 200
    max_pdf_text_chars: int = 2_000_000
    latex_compile_timeout_seconds: float = 120.0
    max_latex_chars: int = 200_000

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.arxiv_max_results <= 0:
            raise ValueError("ARXIV_MAX_RESULTS must be greater than zero")
        if self.arxiv_max_results > self.max_search_results:
            raise ValueError("ARXIV_MAX_RESULTS cannot exceed MAX_SEARCH_RESULTS")
        if self.session_ttl_minutes <= 0:
            raise ValueError("SESSION_TTL_MINUTES must be greater than zero")
        if self.max_active_sessions <= 0:
            raise ValueError("MAX_ACTIVE_SESSIONS must be greater than zero")
        positive_fields = {
            "HTTP_CONNECT_TIMEOUT_SECONDS": self.http_connect_timeout_seconds,
            "HTTP_READ_TIMEOUT_SECONDS": self.http_read_timeout_seconds,
            "HTTP_MAX_ATTEMPTS": self.http_max_attempts,
            "HTTP_BACKOFF_SECONDS": self.http_backoff_seconds,
            "HTTP_MAX_RETRY_DELAY_SECONDS": self.http_max_retry_delay_seconds,
            "LLM_TIMEOUT_SECONDS": self.llm_timeout_seconds,
            "MAX_SEARCH_QUERY_CHARS": self.max_search_query_chars,
            "MAX_SEARCH_RESULTS": self.max_search_results,
            "MAX_CHAT_MESSAGE_CHARS": self.max_chat_message_chars,
            "MAX_PAPER_TITLE_CHARS": self.max_paper_title_chars,
            "MAX_RETRIEVAL_K": self.max_retrieval_k,
            "MAX_PDF_BYTES": self.max_pdf_bytes,
            "MAX_PDF_PAGES": self.max_pdf_pages,
            "MAX_PDF_TEXT_CHARS": self.max_pdf_text_chars,
            "LATEX_COMPILE_TIMEOUT_SECONDS": self.latex_compile_timeout_seconds,
            "MAX_LATEX_CHARS": self.max_latex_chars,
            "CACHE_TTL_SECONDS": self.cache_ttl_seconds,
            "REDIS_CONNECT_TIMEOUT_SECONDS": self.redis_connect_timeout_seconds,
            "REDIS_SOCKET_TIMEOUT_SECONDS": self.redis_socket_timeout_seconds,
            "DATABASE_POOL_SIZE": self.database_pool_size,
            "DATABASE_POOL_TIMEOUT_SECONDS": self.database_pool_timeout_seconds,
            "DATABASE_CONNECT_TIMEOUT_SECONDS": self.database_connect_timeout_seconds,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.http_jitter_seconds < 0:
            raise ValueError("HTTP_JITTER_SECONDS cannot be negative")
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES cannot be negative")
        if self.database_max_overflow < 0:
            raise ValueError("DATABASE_MAX_OVERFLOW cannot be negative")
        if not 0.0 <= self.langsmith_tracing_sampling_rate <= 1.0:
            raise ValueError(
                "LANGSMITH_TRACING_SAMPLING_RATE must be between zero and one"
            )
        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError(
                "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled"
            )
        if not self.cache_key_prefix.strip():
            raise ValueError("CACHE_KEY_PREFIX cannot be empty")
        if self.cache_enabled and self.redis_url is None:
            raise ValueError("REDIS_URL is required when CACHE_ENABLED is enabled")
        if self.database_enabled and self.database_url is None:
            raise ValueError("DATABASE_URL is required when DATABASE_ENABLED is enabled")
        if self.database_url is not None:
            database_url = self.database_url.get_secret_value()
            if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
                raise ValueError("DATABASE_URL must use PostgreSQL")
        return self

    def require_google_api_key(self) -> str:
        """Return the Gemini key or fail with a safe configuration error."""
        if self.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY is not configured")
        return self.google_api_key.get_secret_value()

    def get_semantic_scholar_api_key(self) -> str | None:
        """Return the optional Semantic Scholar key without exposing it in reprs."""
        if self.semantic_scholar_api_key is None:
            return None
        return self.semantic_scholar_api_key.get_secret_value()

    def get_langsmith_api_key(self) -> str | None:
        """Return the optional LangSmith key without exposing it in reprs."""
        if self.langsmith_api_key is None:
            return None
        return self.langsmith_api_key.get_secret_value()

    def get_redis_url(self) -> str | None:
        """Return the optional Redis URL without exposing credentials in reprs."""
        if self.redis_url is None:
            return None
        return self.redis_url.get_secret_value()

    def get_database_url(self) -> str | None:
        """Return the optional PostgreSQL URL without exposing it in reprs."""
        if self.database_url is None:
            return None
        return self.database_url.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable configuration snapshot."""
    return Settings()


def get_llm(api_key: str | None = None) -> ChatGoogleGenerativeAI:
    """Create an LLM client using a session override or server configuration."""
    settings = get_settings()
    resolved_api_key = api_key.strip() if api_key else settings.require_google_api_key()
    if not resolved_api_key:
        raise ValueError("GOOGLE_API_KEY is not configured")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=resolved_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


# Backwards-compatible constants for modules that use these as import-time defaults.
_settings = get_settings()
LLM_MODEL = _settings.llm_model
LLM_TEMPERATURE = _settings.llm_temperature
CHUNK_SIZE = _settings.chunk_size
CHUNK_OVERLAP = _settings.chunk_overlap
EMBEDDING_MODEL = _settings.embedding_model
ARXIV_MAX_RESULTS = _settings.arxiv_max_results
MAX_SEARCH_QUERY_CHARS = _settings.max_search_query_chars
MAX_SEARCH_RESULTS = _settings.max_search_results
MAX_CHAT_MESSAGE_CHARS = _settings.max_chat_message_chars
MAX_PAPER_TITLE_CHARS = _settings.max_paper_title_chars
MAX_RETRIEVAL_K = _settings.max_retrieval_k
MAX_PDF_BYTES = _settings.max_pdf_bytes
MAX_PDF_PAGES = _settings.max_pdf_pages
MAX_PDF_TEXT_CHARS = _settings.max_pdf_text_chars
LATEX_COMPILE_TIMEOUT_SECONDS = _settings.latex_compile_timeout_seconds
MAX_LATEX_CHARS = _settings.max_latex_chars
THREAD_ID = "research-session-001"
