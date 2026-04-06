"""Abstract base class for all occlusion/contamination models.

Formalizes the duck-typed interface shared by AdversarialDustModel,
DynamicBlobDustModel, FingerprintSmudgeModel, AdversarialGlareModel,
and RainOcclusionModel.  Introduces ContaminationMode to distinguish
image-space post-processing models from shader-based (Isaac Sim) models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import numpy as np


class ContaminationMode(Enum):
    """How the occlusion model injects contamination into the pipeline."""

    POST_PROCESS = "post_process"
    """Model operates on rendered images via apply()."""

    SHADER = "shader"
    """Model provides renderer shader parameters via get_shader_params()."""


class OcclusionModel(ABC):
    """Abstract base for CMA-ES-optimizable camera occlusion models.

    Every concrete model must expose:
      - ``n_params``        – dimensionality of the CMA-ES search space
      - ``budget_level``    – pixel-coverage budget constraint
      - ``apply()``         – render contamination onto an image
      - ``get_alpha_mask()``– return the coverage mask for a given param vector
      - ``compute_coverage()`` – fraction of dirty pixels
      - ``get_cma_bounds()``– lower/upper bounds for CMA-ES
      - ``get_cma_x0()``   – initial CMA-ES centroid
      - ``get_random_params()`` – sample a random param vector within budget

    Shader-mode models additionally override ``get_shader_params()``.
    """

    n_params: int
    budget_level: float
    contamination_mode: ContaminationMode = ContaminationMode.POST_PROCESS

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Apply occlusion to *image* given *params* at *timestep*.

        For POST_PROCESS models this performs the full rendering pipeline.
        For SHADER models this is a no-op (returns image unchanged).
        """
        ...

    @abstractmethod
    def get_alpha_mask(
        self, params: np.ndarray, timestep: int = 0
    ) -> np.ndarray:
        """Return the 2-D coverage/attenuation mask for *params*."""
        ...

    @abstractmethod
    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels above the dirty threshold."""
        ...

    @abstractmethod
    def get_cma_bounds(self) -> Tuple[list, list]:
        """Return ``(lower_bounds, upper_bounds)`` arrays for CMA-ES."""
        ...

    @abstractmethod
    def get_cma_x0(self) -> np.ndarray:
        """Return a reasonable initial centroid for CMA-ES."""
        ...

    @abstractmethod
    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random parameter vector respecting the budget."""
        ...

    # ------------------------------------------------------------------
    # Shader-mode hook (concrete default raises)
    # ------------------------------------------------------------------

    def get_shader_params(
        self, params: np.ndarray, timestep: int = 0
    ) -> dict:
        """Return a dict of renderer shader uniforms for *params*.

        Only meaningful when ``contamination_mode is SHADER``.
        POST_PROCESS models raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses POST_PROCESS mode; "
            "get_shader_params() is not supported."
        )
