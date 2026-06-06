from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import UUID, uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from backend.logging_config import get_logger

REQUEST_ID_HEADER = "X-Request-ID"


def _coerce_request_id(raw: str | None) -> UUID:
    if raw:
        try:
            return UUID(raw)
        except ValueError:
            pass
    return uuid4()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request_id, bind it into structlog context, echo it in headers."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._log = get_logger("backend.http")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = _coerce_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = rid
        bound = structlog.contextvars.bind_contextvars(
            request_id=str(rid),
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.reset_contextvars(**bound)
        response.headers[REQUEST_ID_HEADER] = str(rid)
        return response
