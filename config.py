"""
config.py
─────────
Central configuration module for Sentinel AI Gateway.

All environment variables are loaded once at startup via pydantic-settings,
validated, and exposed as a singleton `settings` object that every other module
imports.  Never read `os.environ` directly elsewhere – always use `settings`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Validated application settings loaded from environment variables / .env file.

    Section: OpenAI
    """

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    openai_default_model: str = Field(
        default="gpt-4o", alias="OPENAI_DEFAULT_MODEL"
    )

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com", alias="ANTHROPIC_BASE_URL"
    )
    anthropic_default_model: str = Field(
        default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_DEFAULT_MODEL"
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GEMINI_BASE_URL",
    )
    gemini_default_model: str = Field(
        default="gemini-1.5-pro", alias="GEMINI_DEFAULT_MODEL"
    )

    # ── Crusoe ────────────────────────────────────────────────────────────────
    crusoe_api_key: str = Field(default="", alias="CRUSOE_API_KEY")
    crusoe_base_url: str = Field(
        default="https://api.crusoe.ai/v1", alias="CRUSOE_BASE_URL"
    )
    crusoe_default_model: str = Field(
        default="llama-3-70b-instruct", alias="CRUSOE_DEFAULT_MODEL"
    )

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_default_model: str = Field(
        default="llama3", alias="OLLAMA_DEFAULT_MODEL"
    )

    # ── Gateway behaviour ─────────────────────────────────────────────────────
    provider_priority: List[str] = Field(
        default=["openai", "anthropic", "gemini", "crusoe", "ollama"],
        alias="PROVIDER_PRIORITY",
    )

    # Circuit-breaker thresholds
    cb_failure_threshold: int = Field(default=5, alias="CB_FAILURE_THRESHOLD")
    cb_recovery_timeout: int = Field(default=60, alias="CB_RECOVERY_TIMEOUT")
    cb_success_threshold: int = Field(default=2, alias="CB_SUCCESS_THRESHOLD")

    # Retry settings
    retry_max_attempts: int = Field(default=3, alias="RETRY_MAX_ATTEMPTS")
    retry_initial_wait: float = Field(default=1.0, alias="RETRY_INITIAL_WAIT")
    retry_max_wait: float = Field(default=10.0, alias="RETRY_MAX_WAIT")

    # HTTP timeout (seconds)
    request_timeout: float = Field(default=30.0, alias="REQUEST_TIMEOUT")

    # Application meta
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── pydantic-settings config ──────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        # Allow comma-separated list from env var, e.g. PROVIDER_PRIORITY=openai,gemini
        extra="ignore",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("provider_priority", mode="before")
    @classmethod
    def parse_provider_list(cls, v: object) -> List[str]:
        """Accept either a real list or a comma-separated string from the env."""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v  # type: ignore[return-value]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Usage::

        from config import get_settings
        settings = get_settings()
    """
    return Settings()


# Convenience alias so callers can do `from config import settings`
settings: Settings = get_settings()
