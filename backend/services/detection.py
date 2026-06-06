from __future__ import annotations

import asyncio
import time
from uuid import UUID

import anyio

from detector import ModelRegistry
from shared.contracts import DetectResponse, Diagnostics, ModelRef


class InferenceQueueTimeout(Exception):  # noqa: N818
    """Raised when the inference semaphore could not be acquired in time."""


class DetectorService:
    """Application service that owns the inference concurrency policy.

    Hot path:
        async with registry.use() as detector:                # strong-ref
            await acquire_slot()                              # bounded by semaphore
            outcome = await anyio.to_thread.run_sync(...)     # offload to threadpool

    The semaphore caps how many threads are concurrently inside `predict`.
    Tunes GPU/CPU pressure independently of FastAPI's default thread limiter.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        max_parallel: int,
        queue_timeout_s: float,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._queue_timeout_s = queue_timeout_s
        self._max_parallel = max_parallel

    @property
    def max_parallel(self) -> int:
        return self._max_parallel

    async def detect(self, *, text: str, request_id: UUID) -> DetectResponse:
        async with self._registry.use() as detector:
            start = time.perf_counter()
            await self._acquire_slot()
            try:
                outcome = await anyio.to_thread.run_sync(detector.predict, text)
            finally:
                self._semaphore.release()
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            info = detector.info()
            return DetectResponse(
                verdict=outcome.verdict,
                ai_probability=outcome.ai_probability,
                human_probability=outcome.human_probability,
                confidence=abs(outcome.ai_probability - 0.5) * 2,
                model=ModelRef(name=info.name, version=info.version, device=info.device),
                processing_time_ms=elapsed_ms,
                diagnostics=Diagnostics(
                    tokens=outcome.tokens, truncated=outcome.truncated, chunks=outcome.chunks
                ),
                request_id=request_id,
            )

    async def _acquire_slot(self) -> None:
        if self._queue_timeout_s <= 0:
            await self._semaphore.acquire()
            return
        try:
            async with asyncio.timeout(self._queue_timeout_s):
                await self._semaphore.acquire()
        except TimeoutError as exc:
            raise InferenceQueueTimeout(
                f"backend overloaded: could not acquire inference slot within "
                f"{self._queue_timeout_s:.1f}s"
            ) from exc
