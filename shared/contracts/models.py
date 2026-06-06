from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    loaded: bool
    device: str | None = None
    version: str | None = None
    labels: list[str] = Field(default_factory=list)
    max_input_chars: int | None = Field(default=None, ge=1)


class ModelsListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: str
    available: list[ModelInfo]


class SwitchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class SwitchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active: ModelInfo
