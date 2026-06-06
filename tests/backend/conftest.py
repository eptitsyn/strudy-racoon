from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import create_app
from backend.settings import BackendSettings


def _write_models_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(
        "default: fast\n"
        "models:\n"
        "  - name: fast\n"
        "    impl: stub\n"
        "    params:\n"
        "      bias: 0.3\n"
        "      version: '1.0'\n"
        "  - name: slow\n"
        "    impl: stub\n"
        "    params:\n"
        "      load_delay_ms: 50\n"
        "      bias: -0.3\n"
        "      version: '1.0'\n"
        "  - name: human-leaning\n"
        "    impl: stub\n"
        "    params:\n"
        "      bias: -0.4\n"
        "      version: '1.0'\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def settings(tmp_path: Path) -> BackendSettings:
    return BackendSettings(
        models_config_path=_write_models_yaml(tmp_path),
        detector_max_parallel=4,
        detector_queue_timeout_s=5.0,
        log_level="WARNING",
    )


@pytest.fixture
async def client(settings: BackendSettings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings=settings)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as ac,
        # Trigger lifespan via explicit ASGI startup/shutdown.
        _Lifespan(app),
    ):
        yield ac


class _Lifespan:
    """Minimal helper that drives FastAPI lifespan via the ASGI protocol."""

    def __init__(self, app: object) -> None:
        self._app = app
        self._scope = {"type": "lifespan"}
        self._messages: list[dict[str, object]] = []
        self._send_queue: list[dict[str, object]] = []
        self._task = None

    async def __aenter__(self) -> None:
        import asyncio

        self._inbox: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._outbox: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def receive() -> dict[str, object]:
            return await self._inbox.get()

        async def send(msg: dict[str, object]) -> None:
            await self._outbox.put(msg)

        self._task = asyncio.create_task(self._app(self._scope, receive, send))  # type: ignore[operator]
        await self._inbox.put({"type": "lifespan.startup"})
        msg = await self._outbox.get()
        if msg["type"] != "lifespan.startup.complete":
            raise RuntimeError(f"lifespan startup failed: {msg}")

    async def __aexit__(self, *exc: object) -> None:
        await self._inbox.put({"type": "lifespan.shutdown"})
        msg = await self._outbox.get()
        if msg["type"] != "lifespan.shutdown.complete":
            raise RuntimeError(f"lifespan shutdown failed: {msg}")
        if self._task is not None:
            await self._task
