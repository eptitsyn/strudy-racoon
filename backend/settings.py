from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Runtime configuration for the FastAPI backend.

    Sources (priority high→low): process env → `.env` file → defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    models_config_path: Path = Field(default=Path("models.yaml"))
    detector_max_parallel: int = Field(default=4, ge=1, le=128)
    detector_queue_timeout_s: float = Field(default=10.0, ge=0.0)
    detector_unknown_margin: float = Field(default=0.05, ge=0.0, lt=0.5)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


def get_settings() -> BackendSettings:
    return BackendSettings()
