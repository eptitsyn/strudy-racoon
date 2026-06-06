import pytest

from detector.base import DetectionOutcome, Detector
from detector.config import DetectorSpec
from detector.exceptions import ModelNotRegisteredError
from detector.factory import DetectorFactory, default_factory
from shared.contracts import ModelInfo


class _Dummy(Detector):
    def __init__(self, name: str, params: dict[str, object]) -> None:
        super().__init__(name=name)
        self.params = params

    def load(self) -> None: ...

    def predict(self, text: str) -> DetectionOutcome:
        raise NotImplementedError

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, loaded=False)


def _builder(name: str, params: dict[str, object]) -> Detector:
    return _Dummy(name, params)


class TestDetectorFactory:
    def test_register_and_create(self) -> None:
        f = DetectorFactory()
        f.register("dummy", _builder)
        det = f.create(DetectorSpec(name="x", impl="dummy", params={"foo": 1}))
        assert isinstance(det, _Dummy)
        assert det.name == "x"
        assert det.params == {"foo": 1}

    def test_double_register_rejected(self) -> None:
        f = DetectorFactory()
        f.register("dummy", _builder)
        with pytest.raises(ValueError):
            f.register("dummy", _builder)

    def test_unknown_impl(self) -> None:
        f = DetectorFactory()
        with pytest.raises(ModelNotRegisteredError):
            f.create(DetectorSpec(name="x", impl="missing"))

    def test_default_factory_has_stub(self) -> None:
        f = default_factory()
        assert "stub" in f.known_impls()
