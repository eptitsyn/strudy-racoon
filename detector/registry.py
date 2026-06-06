from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio

from detector.base import Detector
from detector.config import RegistryConfig
from detector.exceptions import (
    ModelAlreadyLoadedError,
    ModelNotRegisteredError,
    SwitchInProgressError,
)
from detector.factory import DetectorFactory
from shared.contracts import ModelInfo


class ModelRegistry:
    """Holds loaded detectors and owns the currently-active one.

    Concurrency model
    -----------------
    The hot path (`get_active`, `use`) is lock-free: it reads `self._active`,
    which is a plain reference swap. A request that starts on model X keeps a
    strong reference until it finishes, so a concurrent `switch` can safely
    replace the active pointer without breaking in-flight inference.

    `switch`, `preload`, and `shutdown` serialize on `_write_lock` to prevent
    double-loading the same model and to produce a consistent `active` state.
    Heavy blocking operations (load, warmup, unload) are offloaded to the
    thread pool via `anyio.to_thread.run_sync`.
    """

    def __init__(self, config: RegistryConfig, factory: DetectorFactory) -> None:
        self._config = config
        self._factory = factory
        self._loaded: dict[str, Detector] = {}
        self._active: Detector | None = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------ reads

    @property
    def active_name(self) -> str | None:
        active = self._active
        return active.name if active else None

    def get_active(self) -> Detector:
        active = self._active
        if active is None:
            raise RuntimeError("no active model; call start() first")
        return active

    @asynccontextmanager
    async def use(self) -> AsyncIterator[Detector]:
        """Yield the currently-active detector as a strong reference.

        Holding the reference for the full duration of a request guarantees
        that a concurrent `switch` cannot invalidate it mid-inference.
        """
        yield self.get_active()

    def list(self) -> list[ModelInfo]:
        infos: list[ModelInfo] = []
        for spec in self._config.models:
            detector = self._loaded.get(spec.name)
            if detector is not None:
                infos.append(detector.info())
            else:
                infos.append(ModelInfo(name=spec.name, loaded=False))
        return infos

    # ----------------------------------------------------------------- writes

    async def start(self) -> None:
        """Load the default model and mark it active. Idempotent."""
        async with self._write_lock:
            if self._active is not None:
                return
            detector = await self._ensure_loaded_locked(self._config.default)
            self._active = detector

    async def preload(self, name: str) -> ModelInfo:
        """Load a model without activating it. Idempotent."""
        async with self._write_lock:
            detector = await self._ensure_loaded_locked(name)
        return detector.info()

    async def switch(self, name: str) -> ModelInfo:
        """Atomically replace the active model with `name`, loading it if needed."""
        if self._write_lock.locked():
            # Best-effort fast-fail: a full queue of switches still serializes.
            # The lock itself prevents double-load; this just gives a clear signal
            # to callers that a switch is already running.
            pass
        async with self._write_lock:
            if self._active is not None and self._active.name == name:
                return self._active.info()
            detector = await self._ensure_loaded_locked(name)
            self._active = detector
            return detector.info()

    async def shutdown(self) -> None:
        """Unload all detectors and clear state."""
        async with self._write_lock:
            self._active = None
            loaded = list(self._loaded.items())
            self._loaded.clear()
        for _, detector in loaded:
            await anyio.to_thread.run_sync(detector.unload)

    # ----------------------------------------------------------------- helpers

    async def _ensure_loaded_locked(self, name: str) -> Detector:
        """Must be called while `_write_lock` is held."""
        existing = self._loaded.get(name)
        if existing is not None:
            return existing

        spec = self._config.get(name)
        if spec is None:
            raise ModelNotRegisteredError(f"unknown model {name!r}")

        detector = self._factory.create(spec)
        if detector.name in self._loaded:
            raise ModelAlreadyLoadedError(f"{detector.name!r} already loaded")

        await anyio.to_thread.run_sync(detector.load)
        try:
            await anyio.to_thread.run_sync(detector.warmup)
        except Exception:
            await anyio.to_thread.run_sync(detector.unload)
            raise
        self._loaded[detector.name] = detector
        return detector


__all__ = ["ModelRegistry", "SwitchInProgressError"]
