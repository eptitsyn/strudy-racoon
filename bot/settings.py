from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    bot_token: str = Field(min_length=1)
    # Base URL of a self-hosted Telegram Bot API server. When unset the bot uses
    # the public api.telegram.org; when set, requests route through this server.
    telegram_api_url: str | None = Field(default=None)
    backend_base_url: str = Field(default="http://localhost:8000")
    backend_timeout_connect_s: float = Field(default=5.0, ge=0.1)
    backend_timeout_read_s: float = Field(default=30.0, ge=0.1)
    backend_retries: int = Field(default=3, ge=0, le=10)
    backend_retry_initial_delay_s: float = Field(default=0.5, ge=0.0)
    backend_retry_max_delay_s: float = Field(default=4.0, ge=0.0)

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    bot_max_input_chars: int = Field(default=8_000, ge=1, le=20_000)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("backend_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")
