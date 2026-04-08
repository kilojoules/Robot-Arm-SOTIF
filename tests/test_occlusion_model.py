"""Tests for the OcclusionModel ABC and ContaminationMode enum."""

import numpy as np
import pytest

from adversarial_dust.occlusion_model import ContaminationMode, OcclusionModel
from adversarial_dust.config import (
    DustGridConfig,
    BlobConfig,
    FingerprintConfig,
    GlareConfig,
)
from adversarial_dust.dust_model import AdversarialDustModel
from adversarial_dust.blob_model import DynamicBlobDustModel
from adversarial_dust.fingerprint_model import FingerprintSmudgeModel
from adversarial_dust.glare_model import AdversarialGlareModel


IMAGE_SHAPE = (256, 256, 3)
BUDGET = 0.20


class TestOcclusionModelABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            OcclusionModel()

    def test_contamination_mode_enum_values(self):
        assert ContaminationMode.POST_PROCESS.value == "post_process"
        assert ContaminationMode.SHADER.value == "shader"


# All concrete POST_PROCESS models share the same contract checks.
_POST_PROCESS_MODELS = [
    ("grid", lambda: AdversarialDustModel(DustGridConfig(), IMAGE_SHAPE, BUDGET)),
    ("blob", lambda: DynamicBlobDustModel(BlobConfig(), IMAGE_SHAPE, BUDGET)),
    ("fingerprint", lambda: FingerprintSmudgeModel(FingerprintConfig(), IMAGE_SHAPE, BUDGET)),
    ("glare", lambda: AdversarialGlareModel(GlareConfig(), IMAGE_SHAPE, BUDGET)),
]


@pytest.mark.parametrize("name,factory", _POST_PROCESS_MODELS, ids=[n for n, _ in _POST_PROCESS_MODELS])
class TestPostProcessModels:
    def test_is_occlusion_model(self, name, factory):
        model = factory()
        assert isinstance(model, OcclusionModel)

    def test_contamination_mode_is_post_process(self, name, factory):
        model = factory()
        assert model.contamination_mode == ContaminationMode.POST_PROCESS

    def test_get_shader_params_raises(self, name, factory):
        model = factory()
        params = model.get_cma_x0()
        with pytest.raises(NotImplementedError):
            model.get_shader_params(params)

    def test_api_surface(self, name, factory):
        """Verify all required methods/attributes exist."""
        model = factory()
        assert isinstance(model.n_params, int) and model.n_params > 0
        assert isinstance(model.budget_level, float)

        # CMA interface
        lb, ub = model.get_cma_bounds()
        assert len(lb) == model.n_params
        assert len(ub) == model.n_params

        x0 = model.get_cma_x0()
        assert x0.shape == (model.n_params,)

        rng = np.random.default_rng(42)
        random_params = model.get_random_params(rng)
        assert random_params.shape == (model.n_params,)

        # Image pipeline
        image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)
        result = model.apply(image, x0, timestep=0)
        assert result.shape == IMAGE_SHAPE
        assert result.dtype == np.uint8

        # Coverage
        alpha = model.get_alpha_mask(x0, timestep=0)
        assert alpha.shape[:2] == IMAGE_SHAPE[:2]
        coverage = model.compute_coverage(alpha)
        assert 0.0 <= coverage <= 1.0


class TestRainOcclusionModel:
    """Rain model is special — test it separately since it needs camera_occlusion."""

    def test_is_occlusion_model(self):
        try:
            from adversarial_dust.rain_model import RainOcclusionModel
        except ImportError:
            pytest.skip("camera_occlusion not available")

        model = RainOcclusionModel(budget_level=0.5, image_shape=(256, 256))
        assert isinstance(model, OcclusionModel)
        assert model.contamination_mode == ContaminationMode.POST_PROCESS

    def test_has_cma_x0(self):
        try:
            from adversarial_dust.rain_model import RainOcclusionModel
        except ImportError:
            pytest.skip("camera_occlusion not available")

        model = RainOcclusionModel(budget_level=0.5, image_shape=(256, 256))
        x0 = model.get_cma_x0()
        assert x0.shape == (model.n_params,)
