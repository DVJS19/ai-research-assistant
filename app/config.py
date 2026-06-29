"""
Central config — all settings loaded from .env via pydantic-settings.
Import this anywhere with: from app.config import settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars
    )

    # ── LLM providers ─────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""

    # ── Vector DB ─────────────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_index_name: str = "research-assistant"
    pinecone_environment: str = "us-east-1-aws"
    pinecone_namespace: str = "dj-multi-agent"

    # ── Reranker ──────────────────────────────────────────────────────────────
    cohere_api_key: str = ""

    # ── Web search ────────────────────────────────────────────────────────────
    brave_search_api_key: str = ""

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant"

    # ── Cache ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Observability ─────────────────────────────────────────────────────────
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "ai-research-assistant"

    # ── Agent budget limits ───────────────────────────────────────────────────
    agent_max_steps: int = 15
    agent_max_tokens: int = 50_000
    agent_max_cost_usd: float = 2.00
    agent_max_seconds: int = 300

    # ── Model selection ───────────────────────────────────────────────────────
    worker_model: str = "claude-haiku-4-5"
    synthesis_model: str = "claude-sonnet-4-6"
    fallback_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# Single instance — import this everywhere
settings = Settings()
