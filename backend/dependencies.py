from __future__ import annotations

from fastapi import Depends, Request

from backend.services import DetectorService
from backend.settings import BackendSettings
from detector import ModelRegistry


def get_settings(request: Request) -> BackendSettings:
    settings: BackendSettings = request.app.state.settings
    return settings


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.registry
    return registry


def get_service(request: Request) -> DetectorService:
    service: DetectorService = request.app.state.detector_service
    return service


SettingsDep = Depends(get_settings)
RegistryDep = Depends(get_registry)
ServiceDep = Depends(get_service)
