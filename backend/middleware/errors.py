from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.logging_config import get_logger
from backend.services.detection import InferenceQueueTimeout
from detector.exceptions import (
    DetectorError,
    DetectorLoadError,
    ModelAlreadyLoadedError,
    ModelNotRegisteredError,
    SwitchInProgressError,
)
from shared.contracts import ErrorResponse

log = get_logger("backend.errors")


def _rid(request: Request) -> UUID | None:
    return getattr(request.state, "request_id", None)


def _json(status: int, code: str, message: str, request_id: UUID | None) -> JSONResponse:
    body = ErrorResponse(code=code, message=message, request_id=request_id)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json(422, "validation_error", _first_error(exc), _rid(request))

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _code_from_status(exc.status_code)
        return _json(exc.status_code, code, str(exc.detail), _rid(request))

    @app.exception_handler(ModelNotRegisteredError)
    async def on_unknown_model(request: Request, exc: ModelNotRegisteredError) -> JSONResponse:
        return _json(404, "model_not_found", str(exc), _rid(request))

    @app.exception_handler(ModelAlreadyLoadedError)
    async def on_already_loaded(
        request: Request, exc: ModelAlreadyLoadedError
    ) -> JSONResponse:
        return _json(409, "model_already_loaded", str(exc), _rid(request))

    @app.exception_handler(SwitchInProgressError)
    async def on_switch_busy(request: Request, exc: SwitchInProgressError) -> JSONResponse:
        return _json(409, "switch_in_progress", str(exc), _rid(request))

    @app.exception_handler(DetectorLoadError)
    async def on_load_fail(request: Request, exc: DetectorLoadError) -> JSONResponse:
        log.error("model_load_failed", error=str(exc))
        return _json(503, "model_load_failed", str(exc), _rid(request))

    @app.exception_handler(InferenceQueueTimeout)
    async def on_queue_timeout(
        request: Request, exc: InferenceQueueTimeout
    ) -> JSONResponse:
        return _json(429, "overloaded", str(exc), _rid(request))

    @app.exception_handler(DetectorError)
    async def on_detector(request: Request, exc: DetectorError) -> JSONResponse:
        log.error("detector_error", error=str(exc), exc_type=type(exc).__name__)
        return _json(500, "detector_error", str(exc), _rid(request))

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error", exc_type=type(exc).__name__)
        return _json(500, "internal_error", "internal server error", _rid(request))


def _first_error(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "request validation failed"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", "invalid")
    return f"{loc}: {msg}" if loc else msg


def _code_from_status(status: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "overloaded",
        503: "unavailable",
    }.get(status, "http_error")
