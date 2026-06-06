from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.logging_config import configure_logging, get_logger
from backend.middleware.errors import register_exception_handlers
from backend.middleware.request_context import RequestContextMiddleware
from backend.routers import api_router
from backend.services.detection import DetectorService
from backend.settings import BackendSettings
from detector import ModelRegistry, load_registry_config
from detector.factory import DetectorFactory, default_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: BackendSettings = app.state.settings
    log = get_logger("backend.lifespan")

    registry: ModelRegistry
    if getattr(app.state, "registry", None) is None:
        factory: DetectorFactory = getattr(app.state, "factory", None) or default_factory()
        config = load_registry_config(settings.models_config_path)
        registry = ModelRegistry(config, factory)
        app.state.registry = registry
    else:
        registry = app.state.registry

    log.info("loading_default_model", name=registry._config.default)
    await registry.start()
    log.info("default_model_ready", name=registry.active_name)

    app.state.detector_service = DetectorService(
        registry,
        max_parallel=settings.detector_max_parallel,
        queue_timeout_s=settings.detector_queue_timeout_s,
    )
    try:
        yield
    finally:
        log.info("shutting_down")
        await registry.shutdown()


def create_app(
    settings: BackendSettings | None = None,
    *,
    registry: ModelRegistry | None = None,
    factory: DetectorFactory | None = None,
) -> FastAPI:
    """Build a FastAPI app. Optional DI hooks exist for tests."""
    settings = settings or BackendSettings()
    configure_logging(settings.log_level, json=settings.log_json)

    app = FastAPI(
        title="AI Text Detector",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.factory = factory

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
