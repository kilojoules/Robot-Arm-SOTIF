"""Rain occlusion model adapter for adversarial_dust evaluator.

Wraps the Rain class from camera_occlusion to match the interface
expected by PolicyEvaluator: apply(image, params, timestep),
get_random_params(rng), n_params.

Used for held-out evaluation — the CNN safety predictor is trained on
fingerprint + glare, then tested on rain to measure generalization.
"""

import numpy as np

_Rain = None


def _get_rain_class():
    global _Rain
    if _Rain is None:
        from camera_occlusion.camera_noise import Rain
        _Rain = Rain
    return _Rain


# Parameter indices
_IDX_DENSITY = 0      # drop density scale [0, 1]
_IDX_RADIUS_MIN = 1   # min radius [3, 15]
_IDX_RADIUS_MAX = 2   # max radius [10, 25]
_IDX_MAGNIFICATION = 3  # lens magnification [0.3, 1.5]
_IDX_BLUR = 4         # blur kernel size [3, 11]
NUM_RAIN_PARAMS = 5


class RainOcclusionModel:
    """Adapter for camera_occlusion Rain effect.

    Maps a flat parameter vector + budget level to Rain constructor args.
    budget_level controls the maximum number of drops (severity cap).
    """

    def __init__(self, budget_level: float, max_drops: int = 150,
                 image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.max_drops = max_drops
        self.image_h = image_shape[0]
        self.image_w = image_shape[1] if len(image_shape) > 1 else image_shape[0]
        self.n_params = NUM_RAIN_PARAMS

        # Param bounds for random sampling / CMA
        self.lower_bounds = np.array([0.0, 3.0, 10.0, 0.3, 3.0])
        self.upper_bounds = np.array([1.0, 15.0, 25.0, 1.5, 11.0])

    def _parse_params(self, params: np.ndarray):
        """Convert flat param vector to Rain constructor kwargs."""
        density = np.clip(params[_IDX_DENSITY], 0.0, 1.0)
        num_drops = max(1, int(density * self.max_drops * self.budget_level))

        radius_min = np.clip(params[_IDX_RADIUS_MIN], 3, 15)
        radius_max = np.clip(params[_IDX_RADIUS_MAX], radius_min + 2, 25)
        magnification = np.clip(params[_IDX_MAGNIFICATION], 0.3, 1.5)
        blur = int(np.clip(params[_IDX_BLUR], 3, 11))
        if blur % 2 == 0:
            blur += 1

        return {
            "num_drops": num_drops,
            "radius_range": (int(radius_min), int(radius_max)),
            "magnification": float(magnification),
            "blur_amount": blur,
        }

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Apply rain occlusion to image.

        Each call generates fresh random drop positions (stateless).
        """
        kwargs = self._parse_params(params)
        rain = _get_rain_class()(**kwargs)
        return rain.apply(image)

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random rain parameters within bounds."""
        return rng.uniform(self.lower_bounds, self.upper_bounds)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Not meaningful for rain (positions are random each call).

        Returns a dummy mask for interface compatibility.
        """
        kwargs = self._parse_params(params)
        # Return a uniform mask proportional to drop count
        coverage = kwargs["num_drops"] / self.max_drops
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def get_cma_bounds(self):
        """Return (lower, upper) bound arrays for CMA-ES."""
        return self.lower_bounds.copy(), self.upper_bounds.copy()
