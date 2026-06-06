"""HuggingFace transformers-based detector.

Imports of `torch` and `transformers` happen lazily inside `load()`, so this
module stays importable in environments that only installed the `bot` extra.

The class is split into pure helpers (`_plan_windows`, `_aggregate_probs`)
that don't need torch, plus the load/predict path that does. Tests exercise
the helpers directly; `load()` + `predict()` integration is skipped when
torch is not available.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from detector.base import DetectionOutcome, Detector
from detector.exceptions import DetectorLoadError
from shared.contracts import ModelInfo, Verdict

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

LabelKey = Literal["ai", "human"]
ChunkStrategy = Literal["truncate", "sliding"]

_DEFAULT_MAX_LENGTH = 512
_DEFAULT_STRIDE = 128
_DEFAULT_UNKNOWN_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class _Window:
    """A single tokenizer window: token ids + attention mask (length == max_length)."""

    input_ids: list[int]
    attention_mask: list[int]


def _plan_windows(
    input_ids: list[int],
    *,
    max_length: int,
    strategy: ChunkStrategy,
    stride: int,
    pad_id: int,
    cls_id: int | None,
    sep_id: int | None,
) -> list[_Window]:
    """Split a bare token sequence (no special tokens) into model-ready windows.

    - For `truncate`: keep the first (max_length - special_count) tokens.
    - For `sliding`: emit overlapping windows of (max_length - special_count)
      content tokens with `stride` overlap, wrapping each in the special
      tokens the model expects and padding to `max_length`.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if stride < 0 or stride >= max_length:
        raise ValueError("stride must be in [0, max_length)")

    has_cls = cls_id is not None
    has_sep = sep_id is not None
    special = int(has_cls) + int(has_sep)
    content_len = max_length - special
    if content_len <= 0:
        raise ValueError("max_length too small for special tokens")

    if not input_ids:
        return [_make_window([], max_length, pad_id, cls_id, sep_id)]

    if strategy == "truncate":
        chunks = [input_ids[:content_len]]
    else:
        step = max(1, content_len - stride)
        chunks = []
        for start in range(0, len(input_ids), step):
            piece = input_ids[start : start + content_len]
            if not piece:
                break
            chunks.append(piece)
            if start + content_len >= len(input_ids):
                break

    return [_make_window(c, max_length, pad_id, cls_id, sep_id) for c in chunks]


def _make_window(
    content: list[int],
    max_length: int,
    pad_id: int,
    cls_id: int | None,
    sep_id: int | None,
) -> _Window:
    ids: list[int] = []
    if cls_id is not None:
        ids.append(cls_id)
    ids.extend(content)
    if sep_id is not None:
        ids.append(sep_id)
    real_len = len(ids)
    if real_len < max_length:
        ids.extend([pad_id] * (max_length - real_len))
    mask = [1] * real_len + [0] * (max_length - real_len)
    return _Window(input_ids=ids, attention_mask=mask)


def _aggregate_probs(per_window: list[tuple[float, float]]) -> tuple[float, float]:
    """Return mean (ai_p, human_p) across windows. Input assumed valid."""
    if not per_window:
        raise ValueError("no windows to aggregate")
    ai = sum(p[0] for p in per_window) / len(per_window)
    human = sum(p[1] for p in per_window) / len(per_window)
    return ai, human


class HFTransformerDetector(Detector):
    def __init__(
        self,
        name: str,
        *,
        repo_id: str,
        label_map: dict[LabelKey, int],
        max_length: int = _DEFAULT_MAX_LENGTH,
        chunk_strategy: ChunkStrategy = "truncate",
        stride: int = _DEFAULT_STRIDE,
        device: str = "auto",
        version: str | None = None,
        unknown_margin: float = _DEFAULT_UNKNOWN_MARGIN,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if set(label_map) != {"ai", "human"}:
            raise ValueError("label_map must have exactly keys {'ai', 'human'}")
        if label_map["ai"] == label_map["human"]:
            raise ValueError("label_map indices must differ")
        if not 0.0 <= unknown_margin < 0.5:
            raise ValueError("unknown_margin must be in [0.0, 0.5)")

        self._repo_id = repo_id
        self._label_map = label_map
        self._max_length = max_length
        self._chunk_strategy: ChunkStrategy = chunk_strategy
        self._stride = stride
        self._requested_device = device
        self._version = version
        self._unknown_margin = unknown_margin
        self._cache_dir = cache_dir

        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None
        self._torch: Any = None
        self._device_str: str | None = None

    @classmethod
    def from_params(cls, name: str, params: dict[str, Any]) -> HFTransformerDetector:
        return cls(name=name, **params)

    def load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise DetectorLoadError(
                "torch/transformers not installed; install the 'ml' extra"
            ) from exc

        device_str = self._resolve_device(torch)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self._repo_id, cache_dir=self._cache_dir
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self._repo_id, cache_dir=self._cache_dir
            )
        except Exception as exc:
            raise DetectorLoadError(
                f"failed to load HF model {self._repo_id!r}: {exc}"
            ) from exc
        model.to(device_str)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device_str = device_str

    def warmup(self) -> None:
        if self._model is None:
            return
        self.predict("warmup")

    def predict(self, text: str) -> DetectionOutcome:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError(f"detector {self.name!r} used before load()")
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        bare_ids = tokenizer.encode(text, add_special_tokens=False)
        windows = _plan_windows(
            bare_ids,
            max_length=self._max_length,
            strategy=self._chunk_strategy,
            stride=self._stride,
            pad_id=int(tokenizer.pad_token_id or 0),
            cls_id=(int(tokenizer.cls_token_id) if tokenizer.cls_token_id is not None else None),
            sep_id=(int(tokenizer.sep_token_id) if tokenizer.sep_token_id is not None else None),
        )

        input_ids = torch.tensor([w.input_ids for w in windows], dtype=torch.long)
        attn = torch.tensor([w.attention_mask for w in windows], dtype=torch.long)
        input_ids = input_ids.to(self._device_str)
        attn = attn.to(self._device_str)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=-1)

        ai_idx = self._label_map["ai"]
        human_idx = self._label_map["human"]
        per_window = [
            (float(probs[i, ai_idx].item()), float(probs[i, human_idx].item()))
            for i in range(probs.shape[0])
        ]
        ai_p, human_p = _aggregate_probs(per_window)

        verdict: Verdict
        if abs(ai_p - 0.5) < self._unknown_margin:
            verdict = "unknown"
        elif ai_p >= human_p:
            verdict = "ai"
        else:
            verdict = "human"

        real_tokens = len(bare_ids)
        truncated = self._chunk_strategy == "truncate" and real_tokens > (
            self._max_length - 2
        )

        return DetectionOutcome(
            ai_probability=ai_p,
            human_probability=human_p,
            verdict=verdict,
            tokens=real_tokens,
            truncated=truncated,
            chunks=len(windows),
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch is not None and self._device_str and self._device_str.startswith("cuda"):
            with contextlib.suppress(Exception):  # best-effort
                self._torch.cuda.empty_cache()
        self._torch = None
        self._device_str = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            loaded=self._model is not None,
            device=self._device_str,
            version=self._version,
            labels=["human", "ai"],
            max_input_chars=None,
        )

    def _resolve_device(self, torch: Any) -> str:
        requested = self._requested_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
                return "mps"
            return "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise DetectorLoadError(f"requested device {requested!r} but CUDA unavailable")
        return requested
