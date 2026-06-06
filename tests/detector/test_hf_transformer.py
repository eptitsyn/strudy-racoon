from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from detector.impl.hf_transformer import (
    HFTransformerDetector,
    _aggregate_probs,
    _plan_windows,
)


class TestPlanWindows:
    def test_empty_input_produces_one_padded_window(self) -> None:
        windows = _plan_windows(
            [],
            max_length=8,
            strategy="truncate",
            stride=0,
            pad_id=0,
            cls_id=101,
            sep_id=102,
        )
        assert len(windows) == 1
        assert windows[0].input_ids[:2] == [101, 102]
        assert sum(windows[0].attention_mask) == 2
        assert len(windows[0].input_ids) == 8

    def test_truncate_keeps_head(self) -> None:
        ids = list(range(1, 100))
        windows = _plan_windows(
            ids,
            max_length=10,
            strategy="truncate",
            stride=0,
            pad_id=0,
            cls_id=101,
            sep_id=102,
        )
        assert len(windows) == 1
        assert windows[0].input_ids[0] == 101
        assert windows[0].input_ids[1:9] == list(range(1, 9))
        assert windows[0].input_ids[9] == 102

    def test_sliding_produces_overlapping_windows(self) -> None:
        ids = list(range(1, 21))
        windows = _plan_windows(
            ids,
            max_length=8,
            strategy="sliding",
            stride=2,
            pad_id=0,
            cls_id=101,
            sep_id=102,
        )
        assert len(windows) > 1
        for w in windows:
            assert len(w.input_ids) == 8
            assert w.input_ids[0] == 101

    def test_stride_validation(self) -> None:
        with pytest.raises(ValueError):
            _plan_windows(
                [1, 2],
                max_length=4,
                strategy="sliding",
                stride=5,
                pad_id=0,
                cls_id=None,
                sep_id=None,
            )

    def test_no_special_tokens(self) -> None:
        ids = [1, 2, 3, 4, 5]
        windows = _plan_windows(
            ids,
            max_length=3,
            strategy="truncate",
            stride=0,
            pad_id=0,
            cls_id=None,
            sep_id=None,
        )
        assert windows[0].input_ids == [1, 2, 3]
        assert windows[0].attention_mask == [1, 1, 1]


class TestAggregate:
    def test_mean(self) -> None:
        ai, human = _aggregate_probs([(0.8, 0.2), (0.4, 0.6), (0.6, 0.4)])
        assert ai == pytest.approx(0.6)
        assert human == pytest.approx(0.4)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _aggregate_probs([])


class TestConstructorValidation:
    def _params(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "repo_id": "x/y",
            "label_map": {"ai": 1, "human": 0},
        }
        base.update(overrides)
        return base

    def test_label_map_keys(self) -> None:
        with pytest.raises(ValueError):
            HFTransformerDetector(
                name="x", **self._params(label_map={"ai": 1, "other": 0})
            )

    def test_label_map_same_index(self) -> None:
        with pytest.raises(ValueError):
            HFTransformerDetector(name="x", **self._params(label_map={"ai": 0, "human": 0}))

    def test_unknown_margin_bounds(self) -> None:
        with pytest.raises(ValueError):
            HFTransformerDetector(name="x", **self._params(unknown_margin=0.5))

    def test_info_before_load(self) -> None:
        det = HFTransformerDetector(
            name="hf",
            repo_id="x/y",
            label_map={"ai": 1, "human": 0},
            version="vtest",
        )
        info = det.info()
        assert info.loaded is False
        assert info.version == "vtest"
        assert info.device is None

    def test_predict_before_load_raises(self) -> None:
        det = HFTransformerDetector(
            name="hf",
            repo_id="x/y",
            label_map={"ai": 1, "human": 0},
        )
        with pytest.raises(RuntimeError):
            det.predict("hello")


class TestPredictWithFakeBackend:
    """Full predict path with tokenizer + torch + model replaced by fakes.

    This does not hit the network or load real weights, but exercises the
    tensor assembly, device placement, softmax aggregation, and verdict logic.
    """

    def test_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_torch, captured = _build_fake_torch_module(
            logits_batch=[[[-1.0, 2.0]]]  # one window, clearly AI-leaning
        )
        fake_tokenizer = _build_fake_tokenizer([10, 20, 30])
        fake_model = MagicMock(return_value=SimpleNamespace(logits=fake_torch.tensor([[[-1.0, 2.0]]], dtype="long")))
        fake_model.to = MagicMock()
        fake_model.eval = MagicMock()

        det = HFTransformerDetector(
            name="hf",
            repo_id="x/y",
            label_map={"ai": 1, "human": 0},
            max_length=8,
            chunk_strategy="truncate",
            stride=2,
            device="cpu",
            unknown_margin=0.0,
        )
        det._tokenizer = fake_tokenizer  # type: ignore[assignment]
        det._model = fake_model  # type: ignore[assignment]
        det._torch = fake_torch
        det._device_str = "cpu"

        outcome = det.predict("anything")
        assert outcome.verdict == "ai"
        assert outcome.ai_probability > outcome.human_probability
        assert outcome.tokens == 3
        assert outcome.chunks == 1
        # sanity: we routed through softmax via our fake
        assert captured["softmax_calls"] == 1


# ---------------------------------------------------------------- fakes -------


def _build_fake_tokenizer(encoded: list[int]) -> Any:
    tok = MagicMock()
    tok.encode = MagicMock(return_value=encoded)
    tok.pad_token_id = 0
    tok.cls_token_id = 101
    tok.sep_token_id = 102
    return tok


def _build_fake_torch_module(logits_batch: list[list[list[float]]]) -> tuple[Any, dict[str, int]]:
    """Minimal torch-shaped module: tensor(), softmax(), inference_mode()."""
    captured = {"softmax_calls": 0}

    class _Tensor:
        def __init__(self, data: Any, dtype: str = "long") -> None:
            self.data = data
            self.dtype = dtype
            self.shape = _shape(data)

        def to(self, _device: str) -> _Tensor:
            return self

        def __getitem__(self, idx: tuple[int, int]) -> _Scalar:
            i, j = idx
            return _Scalar(self.data[i][j])

    class _Scalar:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return self._value

    def _shape(data: Any) -> tuple[int, ...]:
        dims: list[int] = []
        cur = data
        while isinstance(cur, list):
            dims.append(len(cur))
            cur = cur[0] if cur else None
        return tuple(dims)

    def tensor(data: Any, dtype: str = "long") -> _Tensor:
        return _Tensor(data, dtype=dtype)

    def softmax(t: _Tensor, dim: int) -> _Tensor:
        captured["softmax_calls"] += 1
        # data is [[[lo, hi]]] or [[lo, hi]] depending on how logits enter —
        # we produced it in the fake model as [[lo, hi]] (batch=1, labels=2).
        result: list[list[float]] = []
        src = t.data
        if isinstance(src[0][0], list):
            src = src[0]
        for row in src:
            import math

            exps = [math.exp(x) for x in row]
            s = sum(exps)
            result.append([e / s for e in exps])
        return _Tensor(result, dtype="float")

    class _InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    module = SimpleNamespace(
        tensor=tensor,
        softmax=softmax,
        long="long",
        inference_mode=_InferenceMode,
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    _ = logits_batch  # kept for signature symmetry; logits provided via fake model
    return module, captured
