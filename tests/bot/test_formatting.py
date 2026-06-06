from uuid import UUID

from bot.formatting import (
    empty_message,
    format_detection,
    format_models,
    overloaded_message,
    too_long_message,
    unavailable_message,
    validation_error_message,
)
from shared.contracts import DetectResponse, Diagnostics, ModelRef


def _response(
    verdict: str = "ai",
    ai_p: float = 0.9,
    truncated: bool = False,
    chunks: int = 1,
) -> DetectResponse:
    return DetectResponse(
        verdict=verdict,  # type: ignore[arg-type]
        ai_probability=ai_p,
        human_probability=1 - ai_p,
        confidence=abs(ai_p - 0.5) * 2,
        model=ModelRef(name="stub", version="1", device="cpu"),
        processing_time_ms=12,
        diagnostics=Diagnostics(tokens=15, truncated=truncated, chunks=chunks),
        request_id=UUID("11111111-1111-1111-1111-111111111111"),
    )


class TestFormatDetection:
    def test_ai_verdict(self) -> None:
        out = format_detection(_response("ai", 0.87))
        assert "ИИ" in out
        assert "87.0%" in out
        assert "stub" in out

    def test_human_verdict(self) -> None:
        out = format_detection(_response("human", 0.15))
        assert "человека" in out
        assert "15.0%" in out

    def test_unknown_verdict(self) -> None:
        out = format_detection(_response("unknown", 0.5))
        assert "уверен" in out.lower()

    def test_truncated_warning(self) -> None:
        out = format_detection(_response(truncated=True))
        assert "обрезан" in out

    def test_chunks_shown_when_many(self) -> None:
        assert "Окон: 3" in format_detection(_response(chunks=3))
        assert "Окон" not in format_detection(_response(chunks=1))


class TestMisc:
    def test_models_list(self) -> None:
        out = format_models("stub", ["stub", "other"])
        assert "→ stub" in out
        assert "  other" in out

    def test_messages(self) -> None:
        assert "слишком длинный" in too_long_message(100)
        assert empty_message()
        assert "перегружен" in overloaded_message()
        assert "недоступен" in unavailable_message()
        assert "обработать" in validation_error_message("boom")
