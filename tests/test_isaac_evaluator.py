"""Tests for Isaac Sim evaluators and shader adapters (no GPU needed).

All tests use mock objects that simulate Isaac Sim's vectorized
environment and shader interface without requiring NVIDIA hardware.
"""

import numpy as np
import pytest

from adversarial_dust.config import (
    FingerprintConfig,
    GlareConfig,
    IsaacSimConfig,
    LensContaminationConfig,
)
from adversarial_dust.evaluator import PolicyEvaluator
from adversarial_dust.isaac_sim.shader_adapter import (
    IsaacFingerprintAdapter,
    IsaacGlareAdapter,
)
from adversarial_dust.isaac_sim.lens_contamination_model import (
    LensContaminationModel,
)
from adversarial_dust.isaac_sim.isaac_evaluator import (
    IsaacBatchEvaluator,
    IsaacSimEvaluator,
)
from adversarial_dust.occlusion_model import ContaminationMode, OcclusionModel


IMAGE_SHAPE = (256, 256, 3)


# -----------------------------------------------------------------------
# Mock objects
# -----------------------------------------------------------------------


class MockPolicy:
    def reset(self, task_description):
        pass

    def step(self, image, instruction=None):
        raw = {"world_vector": np.zeros(3), "rotation_delta": np.zeros(3)}
        action = {
            "world_vector": np.zeros(3),
            "rot_axangle": np.zeros(3),
            "gripper": np.array([0.0]),
        }
        return raw, action


class MockIsaacEnv:
    """Simulates Isaac Sim vectorized environment without GPU."""

    def __init__(self, num_envs=4, succeed_after=3, max_steps=5):
        self.num_envs = num_envs
        self._succeed_after = succeed_after
        self._max_steps = max_steps
        self._step_counts = np.zeros(num_envs, dtype=int)
        self._rng = np.random.default_rng(42)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        for i in env_ids:
            self._step_counts[i] = 0
        return self._make_obs()

    def step(self):
        self._step_counts += 1

    def apply_action(self, action):
        pass

    def apply_batch_actions(self, actions):
        pass

    def is_success(self, env_idx=0):
        return bool(self._step_counts[env_idx] >= self._succeed_after)

    def get_success_flags(self, n):
        return (self._step_counts[:n] >= self._succeed_after).astype(float)

    def get_camera_image(self, env_idx=0):
        return self._rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)

    def get_batch_images(self, n):
        return self._rng.integers(0, 256, (n, *IMAGE_SHAPE), dtype=np.uint8)

    def _make_obs(self):
        return {"rgb": self._rng.integers(0, 256, (self.num_envs, *IMAGE_SHAPE), dtype=np.uint8)}


class MockShaderInterface:
    """Records shader parameter set calls for assertion."""

    VALID_PREFIXES = {"num_", "centers", "radii", "opacities", "max_", "image_"}

    def __init__(self):
        self.call_log = []
        self.current_params = {}

    def set_param(self, name, value):
        self.call_log.append((name, value))
        self.current_params[name] = value

    def reset(self):
        self.call_log.clear()
        self.current_params.clear()


# -----------------------------------------------------------------------
# Shader adapter tests (pure data, no GPU)
# -----------------------------------------------------------------------


class TestIsaacFingerprintAdapter:
    def test_translate_returns_correct_keys(self):
        config = FingerprintConfig(num_prints=2)
        adapter = IsaacFingerprintAdapter(config, IMAGE_SHAPE)
        params = np.random.uniform(0, 1, size=2 * 8)

        result = adapter.translate(params)

        assert "num_prints" in result
        assert "centers" in result
        assert "opacities" in result
        assert "defocus_sigma" in result
        assert "grease_color" in result
        assert result["num_prints"] == 2

    def test_translate_shapes(self):
        config = FingerprintConfig(num_prints=3)
        adapter = IsaacFingerprintAdapter(config, IMAGE_SHAPE)
        params = np.random.uniform(0, 1, size=3 * 8)

        result = adapter.translate(params)

        assert result["centers"].shape == (3, 2)
        assert result["opacities"].shape == (3,)
        assert result["scales"].shape == (3,)

    def test_values_clipped(self):
        config = FingerprintConfig(num_prints=1, max_opacity=0.5)
        adapter = IsaacFingerprintAdapter(config, IMAGE_SHAPE)
        # Set opacity way above max
        params = np.array([0.5, 0.5, 0.0, 0.15, 5.0, 0.5, 999.0, 0.0])

        result = adapter.translate(params)

        assert result["opacities"][0] <= 0.5


class TestIsaacGlareAdapter:
    def test_translate_returns_correct_keys(self):
        config = GlareConfig()
        adapter = IsaacGlareAdapter(config, IMAGE_SHAPE)
        params = np.random.uniform(0, 1, size=6)

        result = adapter.translate(params)

        assert "source_x" in result
        assert "source_y" in result
        assert "intensity" in result
        assert "streak_angle_deg" in result
        assert "num_streaks" in result

    def test_values_clipped(self):
        config = GlareConfig(max_intensity=0.8)
        adapter = IsaacGlareAdapter(config, IMAGE_SHAPE)
        params = np.array([0.5, 0.5, 999.0, 0.0, 1.0, 1.0])

        result = adapter.translate(params)

        assert result["intensity"] <= 0.8


# -----------------------------------------------------------------------
# Lens contamination model tests
# -----------------------------------------------------------------------


class TestLensContaminationModel:
    def test_is_occlusion_model(self):
        model = LensContaminationModel(
            LensContaminationConfig(), IMAGE_SHAPE, budget_level=0.2
        )
        assert isinstance(model, OcclusionModel)

    def test_shader_mode(self):
        model = LensContaminationModel(
            LensContaminationConfig(), IMAGE_SHAPE, budget_level=0.2
        )
        assert model.contamination_mode == ContaminationMode.SHADER

    def test_apply_is_noop(self):
        model = LensContaminationModel(
            LensContaminationConfig(), IMAGE_SHAPE, budget_level=0.2
        )
        image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)
        params = model.get_cma_x0()
        result = model.apply(image, params)
        np.testing.assert_array_equal(result, image)

    def test_get_shader_params(self):
        config = LensContaminationConfig(num_regions=3)
        model = LensContaminationModel(config, IMAGE_SHAPE, budget_level=0.2)
        params = model.get_cma_x0()

        shader_params = model.get_shader_params(params)

        assert "num_regions" in shader_params
        assert shader_params["num_regions"] == 3
        assert shader_params["centers"].shape == (3, 2)
        assert shader_params["radii"].shape == (3,)
        assert shader_params["opacities"].shape == (3,)

    def test_alpha_mask_shape(self):
        model = LensContaminationModel(
            LensContaminationConfig(), IMAGE_SHAPE, budget_level=0.2
        )
        params = model.get_cma_x0()
        mask = model.get_alpha_mask(params)
        assert mask.shape == (256, 256)

    def test_cma_interface(self):
        config = LensContaminationConfig(num_regions=4)
        model = LensContaminationModel(config, IMAGE_SHAPE, budget_level=0.2)

        assert model.n_params == 4 * 4  # 4 regions * 4 params each
        lb, ub = model.get_cma_bounds()
        assert len(lb) == model.n_params
        x0 = model.get_cma_x0()
        assert x0.shape == (model.n_params,)

        rng = np.random.default_rng(42)
        random_params = model.get_random_params(rng)
        assert random_params.shape == (model.n_params,)


# -----------------------------------------------------------------------
# Evaluator ABC compliance
# -----------------------------------------------------------------------


class TestEvaluatorCompliance:
    def test_isaac_sim_evaluator_is_subclass(self):
        assert issubclass(IsaacSimEvaluator, PolicyEvaluator)

    def test_isaac_batch_evaluator_is_subclass(self):
        assert issubclass(IsaacBatchEvaluator, PolicyEvaluator)

    def test_batch_evaluator_has_batch_evaluate(self):
        assert hasattr(IsaacBatchEvaluator, "batch_evaluate")
