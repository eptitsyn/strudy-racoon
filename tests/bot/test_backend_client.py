from uuid import UUID

import httpx
import pytest
from pytest_httpx import HTTPXMock

from bot.services import (
    BackendClient,
    BackendUnavailableError,
    BackendValidationError,
)

BASE = "http://backend.test"


def _ok_detect_body(model: str = "stub") -> dict[str, object]:
    return {
        "verdict": "ai",
        "ai_probability": 0.9,
        "human_probability": 0.1,
        "confidence": 0.8,
        "model": {"name": model, "version": "1", "device": "cpu"},
        "processing_time_ms": 12,
        "diagnostics": {"tokens": 10, "truncated": False, "chunks": 1},
        "request_id": "11111111-1111-1111-1111-111111111111",
    }


@pytest.fixture
async def client() -> BackendClient:
    return BackendClient(
        base_url=BASE,
        timeout_connect_s=1.0,
        timeout_read_s=1.0,
        retries=2,
        retry_initial_delay_s=0.0,
        retry_max_delay_s=0.0,
    )


class TestDetect:
    async def test_success(self, client: BackendClient, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=f"{BASE}/v1/detect", json=_ok_detect_body())
        r = await client.detect("hello")
        assert r.verdict == "ai"
        assert r.model.name == "stub"

    async def test_request_id_propagated(
        self, client: BackendClient, httpx_mock: HTTPXMock
    ) -> None:
        rid = UUID("22222222-2222-2222-2222-222222222222")
        httpx_mock.add_response(
            url=f"{BASE}/v1/detect",
            match_headers={"X-Request-ID": str(rid)},
            json=_ok_detect_body(),
        )
        await client.detect("hello", request_id=rid)

    async def test_validation_error_4xx(
        self, client: BackendClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/v1/detect",
            status_code=422,
            json={"code": "validation_error", "message": "text too long", "request_id": None},
        )
        with pytest.raises(BackendValidationError) as ei:
            await client.detect("hi")
        assert ei.value.status == 422
        assert ei.value.code == "validation_error"

    async def test_retries_on_5xx_then_succeeds(
        self, client: BackendClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=f"{BASE}/v1/detect", status_code=503, text="boom")
        httpx_mock.add_response(url=f"{BASE}/v1/detect", json=_ok_detect_body())
        r = await client.detect("hi")
        assert r.verdict == "ai"

    async def test_gives_up_after_max_retries(
        self, client: BackendClient, httpx_mock: HTTPXMock
    ) -> None:
        for _ in range(3):  # retries=2 → 3 total attempts
            httpx_mock.add_response(url=f"{BASE}/v1/detect", status_code=502, text="bad")
        with pytest.raises(BackendUnavailableError):
            await client.detect("hi")

    async def test_transport_error_retried(
        self, client: BackendClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_exception(httpx.ConnectError("boom"))
        httpx_mock.add_response(url=f"{BASE}/v1/detect", json=_ok_detect_body())
        r = await client.detect("hi")
        assert r.model.name == "stub"

    async def test_no_retry_on_4xx(
        self, client: BackendClient, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/v1/detect",
            status_code=400,
            json={"code": "bad_request", "message": "nope"},
        )
        with pytest.raises(BackendValidationError):
            await client.detect("hi")
        # If we had retried, the following assert would fail because there'd
        # be no second registered response.


class TestListModels:
    async def test_success(self, client: BackendClient, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url=f"{BASE}/v1/models",
            json={
                "active": "stub",
                "available": [{"name": "stub", "loaded": True, "device": "cpu"}],
            },
        )
        r = await client.list_models()
        assert r.active == "stub"
        assert len(r.available) == 1
