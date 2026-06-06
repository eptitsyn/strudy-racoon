from uuid import uuid4

import pytest
from pydantic import ValidationError

from shared.contracts import (
    DetectRequest,
    DetectResponse,
    Diagnostics,
    ErrorResponse,
    ModelInfo,
    ModelRef,
    ModelsListResponse,
    SwitchRequest,
    SwitchResponse,
)
from shared.contracts.detection import MAX_INPUT_CHARS


class TestDetectRequest:
    def test_minimal_payload(self) -> None:
        req = DetectRequest(text="hello world")
        assert req.text == "hello world"
        assert req.model is None

    def test_with_model_override(self) -> None:
        req = DetectRequest(text="hello", model="stub")
        assert req.model == "stub"

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            DetectRequest(text="")

    def test_rejects_oversize_text(self) -> None:
        with pytest.raises(ValidationError):
            DetectRequest(text="x" * (MAX_INPUT_CHARS + 1))

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            DetectRequest.model_validate({"text": "hi", "surprise": 1})


class TestDetectResponse:
    def _valid_payload(self) -> dict[str, object]:
        return {
            "verdict": "ai",
            "ai_probability": 0.9,
            "human_probability": 0.1,
            "confidence": 0.8,
            "model": {"name": "stub", "version": "0.1", "device": "cpu"},
            "processing_time_ms": 42,
            "diagnostics": {"tokens": 128, "truncated": False, "chunks": 1},
            "request_id": str(uuid4()),
        }

    def test_round_trip(self) -> None:
        payload = self._valid_payload()
        resp = DetectResponse.model_validate(payload)
        restored = DetectResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp

    def test_verdict_must_be_known(self) -> None:
        payload = self._valid_payload()
        payload["verdict"] = "robot"
        with pytest.raises(ValidationError):
            DetectResponse.model_validate(payload)

    def test_probability_bounds(self) -> None:
        payload = self._valid_payload()
        payload["ai_probability"] = 1.5
        with pytest.raises(ValidationError):
            DetectResponse.model_validate(payload)


class TestModelInfo:
    def test_defaults(self) -> None:
        info = ModelInfo(name="stub", loaded=True)
        assert info.labels == []
        assert info.device is None

    def test_list_response_round_trip(self) -> None:
        resp = ModelsListResponse(
            active="stub",
            available=[
                ModelInfo(name="stub", loaded=True, device="cpu"),
                ModelInfo(name="hf", loaded=False),
            ],
        )
        restored = ModelsListResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp


class TestSwitch:
    def test_request_requires_name(self) -> None:
        with pytest.raises(ValidationError):
            SwitchRequest(name="")

    def test_response_round_trip(self) -> None:
        resp = SwitchResponse(active=ModelInfo(name="stub", loaded=True, device="cpu"))
        restored = SwitchResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp


class TestMisc:
    def test_error_response(self) -> None:
        err = ErrorResponse(code="bad_input", message="empty text", request_id=uuid4())
        restored = ErrorResponse.model_validate_json(err.model_dump_json())
        assert restored == err

    def test_model_ref_optional_fields(self) -> None:
        ref = ModelRef(name="stub")
        assert ref.version is None
        assert ref.device is None

    def test_diagnostics_defaults(self) -> None:
        d = Diagnostics()
        assert d.chunks == 1
        assert d.truncated is False
        assert d.tokens is None
