from __future__ import annotations

from contextlib import suppress
from types import TracebackType
from uuid import UUID, uuid4

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from shared.contracts import (
    DetectRequest,
    DetectResponse,
    ErrorResponse,
    ModelsListResponse,
)


class BackendError(Exception):
    """Generic backend error reported back to the user."""


class BackendUnavailableError(BackendError):
    """Network problem or 5xx — retried automatically up to N attempts."""


class BackendValidationError(BackendError):
    """4xx response — request was bad and won't succeed on retry."""

    def __init__(self, message: str, *, status: int, code: str | None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.RemoteProtocolError):
        return True
    return isinstance(exc, BackendUnavailableError)


class BackendClient:
    """Typed async client for the FastAPI backend.

    - One `httpx.AsyncClient` per bot lifetime, reused across requests.
    - Retries only on network errors and 5xx; 4xx surfaces as `BackendValidationError`.
    - Echoes a `X-Request-ID` header so logs on both sides correlate.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_connect_s: float,
        timeout_read_s: float,
        retries: int,
        retry_initial_delay_s: float,
        retry_max_delay_s: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=timeout_connect_s,
                read=timeout_read_s,
                write=timeout_connect_s,
                pool=timeout_connect_s,
            ),
        )
        self._retries = retries
        self._retry_initial = retry_initial_delay_s
        self._retry_max = retry_max_delay_s

    async def __aenter__(self) -> BackendClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            with suppress(Exception):
                await self._client.aclose()

    # --------------------------------------------------------------- public API

    async def detect(
        self,
        text: str,
        *,
        request_id: UUID | None = None,
    ) -> DetectResponse:
        rid = request_id or uuid4()
        body = DetectRequest(text=text).model_dump(mode="json")
        data = await self._post_json("/v1/detect", body, request_id=rid)
        return DetectResponse.model_validate(data)

    async def list_models(self, *, request_id: UUID | None = None) -> ModelsListResponse:
        rid = request_id or uuid4()
        data = await self._get_json("/v1/models", request_id=rid)
        return ModelsListResponse.model_validate(data)

    # -------------------------------------------------------------- internals

    async def _get_json(self, path: str, *, request_id: UUID) -> dict[str, object]:
        return await self._with_retries(
            lambda: self._client.get(path, headers=self._headers(request_id))
        )

    async def _post_json(
        self,
        path: str,
        body: dict[str, object],
        *,
        request_id: UUID,
    ) -> dict[str, object]:
        return await self._with_retries(
            lambda: self._client.post(path, json=body, headers=self._headers(request_id))
        )

    @staticmethod
    def _headers(request_id: UUID) -> dict[str, str]:
        return {"X-Request-ID": str(request_id)}

    async def _with_retries(self, send: object) -> dict[str, object]:
        attempts = self._retries + 1
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=self._retry_initial, max=self._retry_max
            ),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                response = await send()  # type: ignore[operator]
                return self._parse(response)
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, object]:
        if response.is_success:
            data = response.json()
            assert isinstance(data, dict)
            return data
        if 500 <= response.status_code < 600:
            raise BackendUnavailableError(
                f"backend returned {response.status_code}: {response.text[:200]}"
            )
        # 4xx — surface message + code if present
        message = f"backend returned {response.status_code}"
        code: str | None = None
        try:
            err = ErrorResponse.model_validate(response.json())
            message = err.message
            code = err.code
        except Exception:
            pass
        raise BackendValidationError(message, status=response.status_code, code=code)
