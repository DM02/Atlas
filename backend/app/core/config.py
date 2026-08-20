from functools import lru_cache
from pathlib import Path

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

_INSECURE_JWT_SECRET_DEFAULT = "dev-insecure-secret-change-me-in-production-please-32bytes"

# .env lives at the repo root (docker-compose's env_file: ../.env, and the
# documented "cp .env.example .env" setup step both assume this), not in
# backend/ — resolved as an absolute path from this file's own location
# rather than left relative to "." so `cd backend && pytest` (the documented
# local test invocation) finds it too, not just `docker compose up` (which
# never needed this: it injects real OS env vars directly, which
# pydantic-settings always prefers over the .env file regardless of path).
_REPO_ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Atlas"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://atlas:atlas@localhost:5432/atlas"

    openai_api_key: str | None = None
    openai_embedding_dimension: int = 1536

    # Both LLM and embeddings go through OpenRouter by default — its API is
    # OpenAI-SDK-compatible for chat completions AND (as of testing this in
    # Phase 6, mid-2026 — later than this project's original design docs
    # assumed) for embeddings too. This exists because this project's OpenAI
    # account has no billing credits. Unlike LLM models, OpenRouter has no
    # free-tier embedding model — embedding_model routes to the real OpenAI text-embedding-3-small
    # under the hood, billed in small real amounts against the OpenRouter
    # balance rather than the (empty) OpenAI one. Its 1536-dim output is
    # identical to calling OpenAI directly, so DocumentChunk.embedding's
    # column needs no migration.
    #
    # To go back to calling OpenAI directly for either: set the matching
    # *_base_url to None AND change the matching *_model back to the bare
    # (non-"openai/"-prefixed) OpenAI model name — the two must be changed
    # together, this isn't auto-detected.
    openrouter_api_key: str | None = None
    llm_base_url: str | None = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-20b:free"
    embedding_base_url: str | None = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"

    # http://localhost:5173 is the frontend dev server (direct access);
    # http://localhost is the Phase 7 Nginx entrypoint (infra/nginx/nginx.conf,
    # port 80) — both are real, working access paths (see infra/nginx/nginx.conf).
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost"]

    storage_path: str = "./data/documents"
    max_upload_size_mb: int = 20
    allowed_upload_extensions: list[str] = [".pdf", ".txt", ".docx"]

    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 5

    # Phase 2 additions — both default OFF until Phase 5 evaluation shows they're
    # worth the extra latency (docs/ARCHITECTURE.md §9/§10). Toggle per-request-cycle
    # via env for benchmarking, not a permanent code fork.
    enable_hybrid_search: bool = False
    enable_reranking: bool = False
    retrieval_candidate_pool_size: int = 20
    rrf_k: int = 60
    reranker_model: str = "BAAI/bge-reranker-base"

    # Phase 3: auth. jwt_secret_key has an obviously-insecure default so local dev
    # works out of the box — MUST be overridden via env for any real deployment.
    # No refresh tokens (deliberate simplification, see docs/ARCHITECTURE.md §9):
    # the access token just expires and the user logs in again.
    jwt_secret_key: str = _INSECURE_JWT_SECRET_DEFAULT
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Phase 4: ingestion moves off the request path onto an Arq worker via this queue.
    redis_url: str = "redis://localhost:6379"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def warn_if_insecure_defaults(settings: Settings) -> None:
    """Loudly logs (doesn't refuse to start — this is a portfolio project, not
    a real deployment gate) if the obviously-insecure JWT secret default is
    still active outside `environment=development`. Called once at app
    startup (main.py); doesn't run at every `get_settings()` call since
    that's `@lru_cache`d and this needs to run regardless of cache state.
    """
    is_insecure = settings.jwt_secret_key == _INSECURE_JWT_SECRET_DEFAULT
    if settings.environment != "development" and is_insecure:
        logger.warning(
            "insecure_jwt_secret_key_in_non_development_environment",
            environment=settings.environment,
            fix="Set JWT_SECRET_KEY to a real random value via env — see .env.example",
        )
