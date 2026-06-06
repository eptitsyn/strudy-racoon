from __future__ import annotations

import hashlib
import time
from typing import Any

from detector.base import DetectionOutcome, Detector
from shared.contracts import ModelInfo, Verdict

_DEFAULT_UNKNOWN_MARGIN = 0.05


class StubDetector(Detector):
    """Deterministic, dependency-free detector for tests and local development.

    The "score" is derived from a SHA-256 of the input text, so the result is
    reproducible across runs and platforms. `load_delay_ms` and `predict_delay_ms`
    let tests simulate slow load/inference for concurrency and switch tests.
    """

    def __init__(
        self,
        name: str,
        *,
        load_delay_ms: int = 0,
        predict_delay_ms: int = 0,
        unknown_margin: float = _DEFAULT_UNKNOWN_MARGIN,
        bias: float = 0.0,
        version: str = "0.1.0",
    ) -> None:
        super().__init__(name=name)
        if not 0.0 <= unknown_margin < 0.5:
            raise ValueError("unknown_margin must be in [0.0, 0.5)")
        if not -0.5 <= bias <= 0.5:
            raise ValueError("bias must be in [-0.5, 0.5]")
        self._load_delay_s = max(0, load_delay_ms) / 1000.0
        self._predict_delay_s = max(0, predict_delay_ms) / 1000.0
        self._unknown_margin = unknown_margin
        self._bias = bias
        self._version = version
        self._loaded = False

    @classmethod
    def from_params(cls, name: str, params: dict[str, Any]) -> StubDetector:
        return cls(name=name, **params)

    def load(self) -> None:
        if self._load_delay_s:
            time.sleep(self._load_delay_s)
        self._loaded = True

    def predict(self, text: str) -> DetectionOutcome:
        if not self._loaded:
            raise RuntimeError(f"detector {self.name!r} used before load()")
        if self._predict_delay_s:
            time.sleep(self._predict_delay_s)

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        ai_p = min(max(raw + self._bias, 0.0), 1.0)
        human_p = 1.0 - ai_p

        verdict: Verdict
        if abs(ai_p - 0.5) < self._unknown_margin:
            verdict = "unknown"
        elif ai_p >= 0.5:
            verdict = "ai"
        else:
            verdict = "human"

        return DetectionOutcome(
            ai_probability=ai_p,
            human_probability=human_p,
            verdict=verdict,
            tokens=len(text.split()),
            truncated=False,
            chunks=1,
        )

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            loaded=self._loaded,
            device="cpu",
            version=self._version,
            labels=["human", "ai"],
            max_input_chars=None,
        )
