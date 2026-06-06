from pathlib import Path

import pytest
from pydantic import ValidationError

from detector.config import DetectorSpec, RegistryConfig, load_registry_config


class TestDetectorSpec:
    def test_minimal(self) -> None:
        spec = DetectorSpec(name="stub", impl="stub")
        assert spec.params == {}

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            DetectorSpec.model_validate({"name": "x", "impl": "stub", "junk": 1})


class TestRegistryConfig:
    def test_basic(self) -> None:
        cfg = RegistryConfig(
            default="stub",
            models=[DetectorSpec(name="stub", impl="stub")],
        )
        assert cfg.get("stub") is not None
        assert cfg.get("missing") is None

    def test_default_must_exist(self) -> None:
        with pytest.raises(ValidationError):
            RegistryConfig(
                default="other",
                models=[DetectorSpec(name="stub", impl="stub")],
            )

    def test_unique_names(self) -> None:
        with pytest.raises(ValidationError):
            RegistryConfig(
                default="stub",
                models=[
                    DetectorSpec(name="stub", impl="stub"),
                    DetectorSpec(name="stub", impl="stub"),
                ],
            )

    def test_requires_at_least_one_model(self) -> None:
        with pytest.raises(ValidationError):
            RegistryConfig(default="stub", models=[])


class TestLoadRegistryConfig:
    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text(
            "default: stub\n"
            "models:\n"
            "  - name: stub\n"
            "    impl: stub\n"
            "    params:\n"
            "      load_delay_ms: 5\n",
            encoding="utf-8",
        )
        cfg = load_registry_config(yaml_path)
        assert cfg.default == "stub"
        spec = cfg.get("stub")
        assert spec is not None
        assert spec.params == {"load_delay_ms": 5}

    def test_rejects_non_mapping_root(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_registry_config(bad)
