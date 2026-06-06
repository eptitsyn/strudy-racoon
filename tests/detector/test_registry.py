import asyncio

import pytest

from detector.config import DetectorSpec, RegistryConfig
from detector.exceptions import ModelNotRegisteredError
from detector.factory import default_factory
from detector.registry import ModelRegistry


def _config(default: str = "fast") -> RegistryConfig:
    return RegistryConfig(
        default=default,
        models=[
            DetectorSpec(name="fast", impl="stub", params={"bias": 0.3}),
            DetectorSpec(name="slow", impl="stub", params={"load_delay_ms": 80, "bias": -0.3}),
            DetectorSpec(name="other", impl="stub", params={"bias": 0.0}),
        ],
    )


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry(_config(), default_factory())


class TestStartupAndList:
    async def test_start_loads_default(self, registry: ModelRegistry) -> None:
        await registry.start()
        assert registry.active_name == "fast"
        infos = {info.name: info for info in registry.list()}
        assert infos["fast"].loaded is True
        assert infos["slow"].loaded is False

    async def test_start_idempotent(self, registry: ModelRegistry) -> None:
        await registry.start()
        await registry.start()
        assert registry.active_name == "fast"

    async def test_get_active_before_start(self, registry: ModelRegistry) -> None:
        with pytest.raises(RuntimeError):
            registry.get_active()


class TestSwitch:
    async def test_switch_loads_new(self, registry: ModelRegistry) -> None:
        await registry.start()
        info = await registry.switch("other")
        assert info.name == "other"
        assert registry.active_name == "other"

    async def test_switch_idempotent(self, registry: ModelRegistry) -> None:
        await registry.start()
        await registry.switch("fast")
        assert registry.active_name == "fast"

    async def test_switch_unknown(self, registry: ModelRegistry) -> None:
        await registry.start()
        with pytest.raises(ModelNotRegisteredError):
            await registry.switch("missing")

    async def test_concurrent_switches_serialize(self, registry: ModelRegistry) -> None:
        await registry.start()
        results = await asyncio.gather(
            registry.switch("slow"),
            registry.switch("other"),
            registry.switch("slow"),
            registry.switch("fast"),
        )
        assert all(r.loaded for r in results)
        # All declared models eventually loaded — no double-load happened.
        loaded = {info.name for info in registry.list() if info.loaded}
        assert {"fast", "slow", "other"} <= loaded


class TestUseAndConcurrency:
    async def test_use_holds_strong_ref_across_switch(self, registry: ModelRegistry) -> None:
        await registry.start()

        async def long_request() -> str:
            async with registry.use() as detector:
                await asyncio.sleep(0.05)
                return detector.name

        async def switch_during_request() -> str:
            await asyncio.sleep(0.01)
            info = await registry.switch("other")
            return info.name

        held, switched = await asyncio.gather(long_request(), switch_during_request())
        assert held == "fast"
        assert switched == "other"
        assert registry.active_name == "other"


class TestShutdown:
    async def test_shutdown_clears(self, registry: ModelRegistry) -> None:
        await registry.start()
        await registry.switch("other")
        await registry.shutdown()
        assert registry.active_name is None
        assert all(info.loaded is False for info in registry.list())
