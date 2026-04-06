"""Dust/mud occlusion model adapter for camera_occlusion library.

Wraps the Dust class from camera_occlusion (specks, scratches, splotches)
to match the OcclusionModel interface. Simulates accumulated dirt, mud,
and debris on a camera lens.

Replaces the synthetic fog model as a more realistic atmospheric/environmental
corruption for robot cameras in dusty/dirty environments.
"""

import numpy as np

from adversarial_dust.occlusion_model import ContaminationMode, OcclusionModel

_Dust = None


def _get_dust_class():
    global _Dust
    if _Dust is None:
        from camera_occlusion.camera_noise import Dust
        _Dust = Dust
    return _Dust


# Parameter indices
_IDX_NUM_SPECKS = 0      # speck count scale [0, 1]
_IDX_SPECK_OPACITY = 1   # speck opacity [0.1, 0.8]
_IDX_SPLOTCH_OPACITY = 2 # splotch opacity [0.05, 0.5]
_IDX_SCRATCH_OPACITY = 3 # scratch opacity [0.1, 0.5]
NUM_DUST_PARAMS = 4


class DustCameraModel(OcclusionModel):
    """Adapter for camera_occlusion Dust effect.

    Simulates specks, scratches, and splotchy haze on the camera lens.
    budget_level scales the maximum number of specks and scratches.
    """

    def __init__(self, budget_level: float, max_specks: int = 3000,
                 max_scratches: int = 15, image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.max_specks = max_specks
        self.max_scratches = max_scratches
        self.image_h = image_shape[0]
        self.image_w = image_shape[1] if len(image_shape) > 1 else image_shape[0]
        self.n_params = NUM_DUST_PARAMS
        self.contamination_mode = ContaminationMode.POST_PROCESS

        self.lower_bounds = np.array([0.0, 0.1, 0.05, 0.1])
        self.upper_bounds = np.array([1.0, 0.8, 0.5, 0.5])

    def _parse_params(self, params: np.ndarray):
        speck_scale = np.clip(params[_IDX_NUM_SPECKS], 0.0, 1.0)
        speck_opacity = np.clip(params[_IDX_SPECK_OPACITY], 0.1, 0.8)
        splotch_opacity = np.clip(params[_IDX_SPLOTCH_OPACITY], 0.05, 0.5)
        scratch_opacity = np.clip(params[_IDX_SCRATCH_OPACITY], 0.1, 0.5)

        num_specks = max(1, int(speck_scale * self.max_specks * self.budget_level))
        num_scratches = max(0, int(speck_scale * self.max_scratches * self.budget_level))

        return {
            "num_specks": num_specks,
            "speck_radius_range": (1, max(2, int(3 + 4 * self.budget_level))),
            "speck_opacity": float(speck_opacity),
            "num_scratches": num_scratches,
            "scratch_length_range": (10, max(15, int(30 + 120 * self.budget_level))),
            "scratch_opacity": float(scratch_opacity),
            "splotch_noise_scale": max(4, int(16 - 8 * self.budget_level)),
            "splotch_opacity": float(splotch_opacity * self.budget_level),
        }

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        kwargs = self._parse_params(params)
        dust = _get_dust_class()(**kwargs)
        return dust.apply(image)

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        kwargs = self._parse_params(params)
        coverage = kwargs["num_specks"] / self.max_specks
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return (self.lower_bounds + self.upper_bounds) / 2.0
