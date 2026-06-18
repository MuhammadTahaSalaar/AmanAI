"""Application settings loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # AWS / Bedrock
    AWS_REGION: str = "us-east-1"
    BEDROCK_GEN_MODEL_ID: str = "us.meta.llama3-3-70b-instruct-v1:0"
    BEDROCK_GEN_MODEL_ID_FALLBACK: str = "us.meta.llama3-1-8b-instruct-v1:0"
    BEDROCK_EMBED_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_RERANK_MODEL_ID: str = "cohere.rerank-v3-5:0"
    BEDROCK_GUARDRAIL_ID: Optional[str] = None
    BEDROCK_GUARDRAIL_VERSION: str = "DRAFT"

    # Retrieval / generation knobs
    EMBED_DIM: int = 1024
    RETRIEVE_K: int = 8
    RERANK_TOP_N: int = 4
    MIN_RERANK_SCORE: float = 0.30
    MAX_INPUT_CHARS: int = 2000
    MAX_UPLOAD_MB: int = 5

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    SUPABASE_SECRET_ARN: str = ""

    # Cognito
    COGNITO_USER_POOL_ID: str = ""
    COGNITO_CLIENT_ID: str = ""
    COGNITO_REGION: str = "us-east-1"

    # App
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
