from __future__ import annotations

from collections.abc import Callable
from typing import Any

from detector.base import Detector
from detector.config import DetectorSpec
from detector.exceptions import ModelNotRegisteredError

DetectorBuilder = Callable[[str, dict[str, Any]], Detector]


class DetectorFactory:
    """Maps `impl` strings from config to concrete `Detector` subclasses.

    Decoupling implementations from the registry keeps the ML module open for
    extension: adding a new backend is one builder registration plus a YAML entry.
    """

    def __init__(self) -> None:
        self._builders: dict[str, DetectorBuilder] = {}

    def register(self, impl: str, builder: DetectorBuilder) -> None:
        if impl in self._builders:
            raise ValueError(f"detector impl {impl!r} already registered")
        self._builders[impl] = builder

    def known_impls(self) -> list[str]:
        return sorted(self._builders)

    def create(self, spec: DetectorSpec) -> Detector:
        builder = self._builders.get(spec.impl)
        if builder is None:
            raise ModelNotRegisteredError(
                f"unknown detector impl {spec.impl!r}; known: {self.known_impls()}"
            )
        return builder(spec.name, spec.params)


def default_factory() -> DetectorFactory:
    """Build a factory pre-populated with bundled implementations.

    Note: we import implementation classes lazily so that `stub`-only
    deployments (e.g. the bot's CI suite) don't pay for the optional
    `torch`/`transformers` import even if they never register the HF builder.
    """
    factory = DetectorFactory()

    from detector.impl.stub import StubDetector

    factory.register("stub", StubDetector.from_params)

    try:
        from detector.impl.hf_transformer import HFTransformerDetector

        factory.register("hf_transformer", HFTransformerDetector.from_params)
    except ImportError:
        # hf_transformer.py itself only imports torch lazily, so a bare import
        # should succeed; guard is here for defence in depth.
        pass

    try:
        from detector.impl.pawn_detector import PawnDetector

        factory.register("pawn", PawnDetector.from_params)
    except ImportError:
        # pawn_detector.py imports torch lazily too; guard for defence in depth.
        pass
    return factory
