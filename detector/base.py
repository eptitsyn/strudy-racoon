from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from shared.contracts import ModelInfo, Verdict


@dataclass(frozen=True, slots=True)
class DetectionOutcome:
    """Raw output of a single inference call.

    Probabilities must sum to ~1.0; the registry/service layer is responsible
    for deriving the final `Verdict` (including `unknown` for low-margin cases)
    and packaging into the public `DetectResponse`.
    """

    ai_probability: float
    human_probability: float
    verdict: Verdict
    tokens: int | None = None
    truncated: bool = False
    chunks: int = 1


class Detector(ABC):
    """Abstract detector. Implementations wrap a concrete ML backend.

    Lifecycle: __init__ (cheap, no weights) -> load() -> warmup() ->
    predict()* -> unload(). `predict` is the only hot-path method and is
    expected to be CPU- or GPU-bound; callers MUST run it in a thread
    (anyio.to_thread / asyncio.to_thread). Implementations should not
    introduce additional blocking I/O inside `predict`.
    """

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def load(self) -> None:
        """Materialise weights, move to device, set eval mode. Blocking."""

    def warmup(self) -> None:
        """Optional warmup pass. Default: no-op."""
        return None

    @abstractmethod
    def predict(self, text: str) -> DetectionOutcome:
        """Run inference on a single text. Blocking; must be thread-safe under eval()."""

    def unload(self) -> None:
        """Release resources. Default: no-op (rely on GC)."""
        return None

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return current public-facing metadata."""
