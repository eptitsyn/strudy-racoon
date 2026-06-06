"""PAWN-based detector.

Wraps the standalone PAWN model living under ``detector/impl/pawn/`` (config +
checkpoint in ``detector/impl/pawn/model/``) behind the :class:`Detector`
interface, mirroring :class:`detector.impl.stub.StubDetector`.

The PAWN package modules use top-level (non-package) imports
(``from model import PAWN`` etc.), so ``load()`` puts that directory on
``sys.path`` before importing. ``torch`` is imported lazily inside ``load()``
so this module stays importable in environments without the ``ml`` extra.

Label convention (from training): index 1 == "ai", index 0 == "human", and the
raw logit feeds ``sigmoid`` to give ``P(ai)``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from detector.base import DetectionOutcome, Detector
from detector.exceptions import DetectorLoadError
from shared.contracts import ModelInfo, Verdict

if TYPE_CHECKING:
    from detector.impl.pawn.model import PAWN

_PAWN_DIR = Path(__file__).resolve().parent / "pawn"
_DEFAULT_CONFIG = _PAWN_DIR / "model" / "config.yaml"
_DEFAULT_CHECKPOINT = _PAWN_DIR / "model" / "pytorch_model.bin"
_DEFAULT_UNKNOWN_MARGIN = 0.05


class PawnDetector(Detector):
    """Detector backed by a trained PAWN model.

    `config_path`/`checkpoint_path` default to the bundled
    ``model/config.yaml`` and ``model/pytorch_model.bin``. `device` follows the
    PAWN auto-detection (cuda -> mps -> cpu) when left as ``None``.
    """

    def __init__(
        self,
        name: str,
        *,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        unknown_margin: float = _DEFAULT_UNKNOWN_MARGIN,
        version: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not 0.0 <= unknown_margin < 0.5:
            raise ValueError("unknown_margin must be in [0.0, 0.5)")

        self._config_path = config_path or str(_DEFAULT_CONFIG)
        self._checkpoint_path = checkpoint_path or str(_DEFAULT_CHECKPOINT)
        self._requested_device = device
        self._unknown_margin = unknown_margin
        self._version = version

        self._model: PAWN | None = None
        self._device_str: str | None = None
        self._max_length: int | None = None

    @classmethod
    def from_params(cls, name: str, params: dict[str, Any]) -> PawnDetector:
        return cls(name=name, **params)

    def load(self) -> None:
        try:
            import torch  # noqa: F401  (ensures the ml extra is present)
        except ImportError as exc:
            raise DetectorLoadError(
                "torch not installed; install the 'ml' extra"
            ) from exc

        # The PAWN package uses top-level imports (e.g. `from model import PAWN`),
        # so its directory must be importable as a path root.
        if str(_PAWN_DIR) not in sys.path:
            sys.path.insert(0, str(_PAWN_DIR))

        try:
            from inference import _resolve_checkpoint, load_model
        except ImportError as exc:
            raise DetectorLoadError(f"failed to import PAWN inference: {exc}") from exc

        try:
            checkpoint = _resolve_checkpoint(self._checkpoint_path)
            model, device = load_model(
                self._config_path, checkpoint, self._requested_device
            )
        except Exception as exc:
            raise DetectorLoadError(f"failed to load PAWN model: {exc}") from exc

        self._model = model
        self._device_str = device
        self._max_length = model.pawn_config.max_length

    def warmup(self) -> None:
        if self._model is None:
            return
        self.predict("warmup")

    def predict(self, text: str) -> DetectionOutcome:
        if self._model is None or self._device_str is None:
            raise RuntimeError(f"detector {self.name!r} used before load()")

        from inference import predict as pawn_predict

        result = pawn_predict(self._model, [text], self._device_str, batch_size=1)[0]

        # sigmoid(logit) == P(ai); the inference helper's "prob_human" key is a
        # misnomer for this label convention, so derive directly from the logit.
        ai_p = 1.0 / (1.0 + math.exp(-float(result["logit"])))
        human_p = 1.0 - ai_p

        verdict: Verdict
        if abs(ai_p - 0.5) < self._unknown_margin:
            verdict = "unknown"
        elif ai_p >= human_p:
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

    def unload(self) -> None:
        self._model = None
        self._device_str = None
        self._max_length = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            loaded=self._model is not None,
            device=self._device_str,
            version=self._version,
            labels=["human", "ai"],
            max_input_chars=None,
        )
