"""Tests for Isaac Sim config extensions and backward compatibility."""

import glob
from pathlib import Path

import pytest

from adversarial_dust.config import (
    CalibrationConfig,
    EnvConfig,
    EnvelopeExperimentConfig,
    IsaacSimConfig,
    LensContaminationConfig,
    load_config,
    load_envelope_config,
)


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# -----------------------------------------------------------------------
# Default value tests
# -----------------------------------------------------------------------


class TestIsaacSimConfigDefaults:
    def test_defaults(self):
        cfg = IsaacSimConfig()
        assert cfg.num_envs == 1
        assert cfg.renderer == "ray_traced"
        assert cfg.lens_model == "pinhole"
        assert cfg.gpu_id == 0
        assert cfg.headless is True
        assert cfg.shader_occlusion is True
        assert cfg.usd_scene_path is None

    def test_calibration_defaults(self):
        cfg = CalibrationConfig()
        assert cfg.reference_data_path is None
        assert cfg.fitting_method == "polynomial"
        assert cfg.fitting_degree == 3
        assert cfg.validation_split == 0.2

    def test_lens_contamination_defaults(self):
        cfg = LensContaminationConfig()
        assert cfg.num_regions == 5
        assert cfg.max_roughness == 0.8
        assert cfg.dirty_threshold == 0.05

    def test_env_config_sim_backend_default(self):
        cfg = EnvConfig()
        assert cfg.sim_backend == "simpler_env"

    def test_envelope_isaac_sim_none_by_default(self):
        cfg = EnvelopeExperimentConfig()
        assert cfg.isaac_sim is None
        assert cfg.calibration is None
        # lens_contamination has a non-None default
        assert isinstance(cfg.lens_contamination, LensContaminationConfig)


# -----------------------------------------------------------------------
# Backward compatibility: every existing YAML must still load
# -----------------------------------------------------------------------


def _envelope_yamls():
    """Legacy envelope configs (excludes Isaac Sim config)."""
    return sorted(
        p for p in CONFIGS_DIR.glob("envelope_*.yaml")
        if "isaac_sim" not in p.name
    )


def _all_yamls():
    return sorted(CONFIGS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_path", _envelope_yamls(), ids=lambda p: p.name)
class TestEnvelopeBackwardCompat:
    def test_loads_without_error(self, yaml_path):
        cfg = load_envelope_config(str(yaml_path))
        assert isinstance(cfg, EnvelopeExperimentConfig)

    def test_isaac_sim_is_none(self, yaml_path):
        """Legacy configs have no isaac_sim block -> field stays None."""
        cfg = load_envelope_config(str(yaml_path))
        assert cfg.isaac_sim is None


class TestIsaacSimYAML:
    def test_loads_with_isaac_sim_populated(self):
        yaml_path = CONFIGS_DIR / "envelope_isaac_sim.yaml"
        cfg = load_envelope_config(str(yaml_path))

        assert cfg.isaac_sim is not None
        assert isinstance(cfg.isaac_sim, IsaacSimConfig)
        assert cfg.isaac_sim.num_envs == 8
        assert cfg.isaac_sim.renderer == "ray_traced"
        assert cfg.env.sim_backend == "isaac_sim"


@pytest.mark.parametrize("yaml_path", _all_yamls(), ids=lambda p: p.name)
class TestAllConfigsLoad:
    def test_loads_as_experiment_config(self, yaml_path):
        """All configs should at least parse via load_config (strict=False)."""
        cfg = load_config(str(yaml_path))
        assert cfg is not None
