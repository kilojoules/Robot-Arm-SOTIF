"""Shader-based lens contamination model for Isaac Sim.

Implements the OcclusionModel interface in SHADER mode: CMA-ES optimizes
the same parameter space as the CPU fingerprint/glare models, but instead
of post-processing images, this model provides Warp kernel uniforms that
the Isaac Sim evaluator injects into the rendering pipeline.

For visualization and budget projection, an approximate alpha mask is
computed analytically (no GPU render required).
"""

import numpy as np

from adversarial_dust.config import LensContaminationConfig
from adversarial_dust.occlusion_model import ContaminationMode, OcclusionModel
from adversarial_dust.isaac_sim.shader_adapter import (
    IsaacFingerprintAdapter,
    IsaacGlareAdapter,
)


# Each contamination region has: cx, cy, radius, opacity = 4 params
_PARAMS_PER_REGION = 4
_IDX_CX = 0
_IDX_CY = 1
_IDX_RADIUS = 2
_IDX_OPACITY = 3


class LensContaminationModel(OcclusionModel):
    """Shader-based lens contamination for CMA-ES optimization.

    This model is used when ``contamination_mode is SHADER``.  The
    ``apply()`` method is a no-op (the contamination is baked into the
    Isaac Sim render by the evaluator); ``get_shader_params()`` returns
    the dict of Warp kernel uniforms.

    The parameter space is a set of circular contamination regions on
    the lens, each with position, radius, and opacity.
    """

    contamination_mode = ContaminationMode.SHADER

    def __init__(
        self,
        config: LensContaminationConfig,
        image_shape: tuple,
        budget_level: float,
    ):
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        self.budget_level = budget_level
        self.num_regions = config.num_regions
        self.n_params = self.num_regions * _PARAMS_PER_REGION

        # Pre-compute pixel coordinate grids (normalized 0-1)
        ys = np.linspace(0, 1, self.image_h, dtype=np.float64)
        xs = np.linspace(0, 1, self.image_w, dtype=np.float64)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse_regions(self, params: np.ndarray) -> list:
        """Parse flat param vector into list of region dicts."""
        params = np.asarray(params, dtype=np.float64)
        regions = []
        for i in range(self.num_regions):
            off = i * _PARAMS_PER_REGION
            p = params[off : off + _PARAMS_PER_REGION]
            regions.append({
                "cx": float(np.clip(p[_IDX_CX], 0.0, 1.0)),
                "cy": float(np.clip(p[_IDX_CY], 0.0, 1.0)),
                "radius": float(np.clip(
                    p[_IDX_RADIUS],
                    self.config.radius_range[0],
                    self.config.radius_range[1],
                )),
                "opacity": float(np.clip(
                    p[_IDX_OPACITY],
                    self.config.opacity_range[0],
                    self.config.opacity_range[1],
                )),
            })
        return regions

    # ------------------------------------------------------------------
    # OcclusionModel interface
    # ------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """No-op: contamination is applied by the Isaac Sim renderer."""
        return image

    def get_shader_params(
        self, params: np.ndarray, timestep: int = 0
    ) -> dict:
        """Return Warp kernel uniform dict for these CMA-ES params."""
        regions = self._parse_regions(params)
        return {
            "num_regions": self.num_regions,
            "centers": np.array(
                [[r["cx"], r["cy"]] for r in regions], dtype=np.float32
            ),
            "radii": np.array(
                [r["radius"] for r in regions], dtype=np.float32
            ),
            "opacities": np.array(
                [r["opacity"] for r in regions], dtype=np.float32
            ),
            "max_roughness": self.config.max_roughness,
            "image_height": self.image_h,
            "image_width": self.image_w,
        }

    def get_alpha_mask(
        self, params: np.ndarray, timestep: int = 0
    ) -> np.ndarray:
        """Approximate alpha mask from circular regions (for budget/viz)."""
        regions = self._parse_regions(params)
        mask = np.zeros((self.image_h, self.image_w), dtype=np.float64)
        for r in regions:
            dist = np.sqrt(
                (self.grid_x - r["cx"]) ** 2
                + (self.grid_y - r["cy"]) ** 2
            )
            region_mask = r["opacity"] * np.clip(
                1.0 - dist / max(r["radius"], 1e-6), 0.0, 1.0
            )
            mask = np.maximum(mask, region_mask)
        return self._project_to_budget(mask)

    def _project_to_budget(self, mask: np.ndarray) -> np.ndarray:
        """Scale mask so coverage <= budget_level via binary search."""
        if self.budget_level >= 1.0:
            return mask

        coverage = self.compute_coverage(mask)
        if coverage <= self.budget_level:
            return mask

        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if self.compute_coverage(mask * mid) <= self.budget_level:
                lo = mid
            else:
                hi = mid
        return mask * lo

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels above dirty threshold."""
        return float(np.mean(alpha_mask > self.config.dirty_threshold))

    # ------------------------------------------------------------------
    # CMA-ES interface
    # ------------------------------------------------------------------

    def get_cma_bounds(self) -> tuple:
        lb, ub = [], []
        for _ in range(self.num_regions):
            lb.extend([0.0, 0.0, self.config.radius_range[0], self.config.opacity_range[0]])
            ub.extend([1.0, 1.0, self.config.radius_range[1], self.config.opacity_range[1]])
        return lb, ub

    def get_cma_x0(self) -> np.ndarray:
        x0 = []
        mid_r = sum(self.config.radius_range) / 2
        mid_o = sum(self.config.opacity_range) / 2
        for _ in range(self.num_regions):
            x0.extend([0.5, 0.5, mid_r, mid_o])
        return np.array(x0, dtype=np.float64)

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        lb, ub = self.get_cma_bounds()
        return rng.uniform(lb, ub)
