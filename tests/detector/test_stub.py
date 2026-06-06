import pytest

from detector.impl.stub import StubDetector


class TestStubDetector:
    def test_predict_requires_load(self) -> None:
        det = StubDetector(name="stub")
        with pytest.raises(RuntimeError):
            det.predict("hello")

    def test_deterministic(self) -> None:
        a = StubDetector(name="stub", unknown_margin=0.0)
        b = StubDetector(name="stub", unknown_margin=0.0)
        a.load()
        b.load()
        assert a.predict("same text").ai_probability == b.predict("same text").ai_probability

    def test_probabilities_sum_to_one(self) -> None:
        det = StubDetector(name="stub", unknown_margin=0.0)
        det.load()
        out = det.predict("anything")
        assert out.ai_probability + out.human_probability == pytest.approx(1.0)

    def test_unknown_zone(self) -> None:
        det = StubDetector(name="stub", unknown_margin=0.49, bias=0.0)
        det.load()
        out = det.predict("borderline" * 3)
        assert out.verdict == "unknown"

    def test_bias_pushes_to_ai(self) -> None:
        det = StubDetector(name="stub", unknown_margin=0.0, bias=0.5)
        det.load()
        out = det.predict("neutral")
        assert out.verdict == "ai"
        assert out.ai_probability >= 0.5

    def test_info_reflects_load_state(self) -> None:
        det = StubDetector(name="stub")
        assert det.info().loaded is False
        det.load()
        assert det.info().loaded is True

    def test_invalid_unknown_margin(self) -> None:
        with pytest.raises(ValueError):
            StubDetector(name="stub", unknown_margin=0.5)

    def test_from_params(self) -> None:
        det = StubDetector.from_params("s", {"bias": 0.1, "version": "9"})
        assert det.name == "s"
        assert det.info().version == "9"
