from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DetectorSpec(BaseModel):
    """Declarative description of a registrable detector.

    `impl` is a string key resolved by `DetectorFactory` to a concrete class.
    `params` is the kwargs blob passed to that class's `__init__` (besides `name`).
    Validation of `params` content is deferred to the implementation, since each
    backend has its own required fields.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(min_length=1)
    impl: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class RegistryConfig(BaseModel):
    """Top-level registry configuration: list of declared models + which is default."""

    model_config = ConfigDict(extra="forbid")

    default: str = Field(min_length=1)
    models: list[DetectorSpec] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _unique_names(cls, value: list[DetectorSpec]) -> list[DetectorSpec]:
        names = [m.name for m in value]
        if len(set(names)) != len(names):
            raise ValueError("model names in registry must be unique")
        return value

    def get(self, name: str) -> DetectorSpec | None:
        for spec in self.models:
            if spec.name == name:
                return spec
        return None

    def model_post_init(self, _: Any) -> None:
        if self.get(self.default) is None:
            raise ValueError(f"default model {self.default!r} is not in models[]")


def load_registry_config(path: str | Path) -> RegistryConfig:
    """Read a YAML registry config from disk."""
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return RegistryConfig.model_validate(data)
