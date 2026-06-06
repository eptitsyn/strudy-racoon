from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["ai", "human", "unknown"]

MAX_INPUT_CHARS = 20_000


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    version: str | None = None
    device: str | None = None


class Diagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: int | None = Field(default=None, ge=0)
    truncated: bool = False
    chunks: int = Field(default=1, ge=1)


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    model: str | None = None


class DetectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    verdict: Verdict
    ai_probability: float = Field(ge=0.0, le=1.0)
    human_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model: ModelRef
    processing_time_ms: int = Field(ge=0)
    diagnostics: Diagnostics
    request_id: UUID
