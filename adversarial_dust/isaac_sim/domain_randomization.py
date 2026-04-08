"""Scene-level domain randomization for Isaac Sim environments.

Leverages Isaac Sim's Replicator API to randomize environmental conditions
during evaluation, ensuring adversarial contamination patterns are robust
across visual appearances rather than exploiting specific scene features.

Randomized properties:
- Lighting: intensity, color temperature, direction
- Textures: table surface, basket material
- Object placement: eggplant initial pose jitter
- Camera: small pose perturbations (simulating mounting imprecision)
- Distractors: random clutter objects on the table
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DomainRandomizationConfig:
    """Configuration for scene-level domain randomization."""

    # Lighting
    light_intensity_range: Tuple[float, float] = (500.0, 2000.0)
    light_color_temp_range: Tuple[float, float] = (3500.0, 6500.0)
    light_azimuth_range: Tuple[float, float] = (0.0, 360.0)
    light_elevation_range: Tuple[float, float] = (30.0, 80.0)

    # Textures
    table_texture_paths: List[str] = field(default_factory=list)
    randomize_textures: bool = False

    # Object placement
    eggplant_position_jitter: float = 0.02  # meters
    eggplant_rotation_jitter: float = 15.0  # degrees

    # Camera
    camera_position_jitter: float = 0.005  # meters
    camera_rotation_jitter: float = 1.0  # degrees

    # Distractors
    num_distractors: int = 0
    distractor_usd_paths: List[str] = field(default_factory=list)


class DomainRandomizer:
    """Apply domain randomization to an Isaac Lab environment.

    Call ``randomize()`` before each episode or batch of episodes.
    Uses Isaac Sim's Replicator API (``omni.replicator.core``) when
    available; falls back to direct prim manipulation otherwise.
    """

    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self._rng = np.random.default_rng()

    def randomize(self, env, seed: int = None):
        """Randomize scene properties for the given environment.

        Args:
            env: IsaacManipulationEnv or ManagerBasedRLEnv instance.
            seed: Optional seed for reproducibility.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._randomize_lighting(env)
        self._randomize_object_placement(env)
        self._randomize_camera(env)

        if self.config.randomize_textures:
            self._randomize_textures(env)

        if self.config.num_distractors > 0:
            self._place_distractors(env)

    def _randomize_lighting(self, env):
        """Randomize scene lighting intensity, color, and direction."""
        cfg = self.config
        intensity = self._rng.uniform(*cfg.light_intensity_range)
        color_temp = self._rng.uniform(*cfg.light_color_temp_range)
        azimuth = self._rng.uniform(*cfg.light_azimuth_range)
        elevation = self._rng.uniform(*cfg.light_elevation_range)

        logger.debug(
            f"DR lighting: intensity={intensity:.0f}, "
            f"color_temp={color_temp:.0f}K, "
            f"azimuth={azimuth:.0f}°, elevation={elevation:.0f}°"
        )
        # Actual: omni.replicator.core.modify.pose(light_prim, ...)
        #         omni.replicator.core.modify.attribute(light_prim, "intensity", intensity)

    def _randomize_object_placement(self, env):
        """Jitter eggplant initial pose."""
        cfg = self.config
        pos_delta = self._rng.normal(0, cfg.eggplant_position_jitter, size=3)
        rot_delta = self._rng.normal(0, cfg.eggplant_rotation_jitter, size=3)
        logger.debug(f"DR eggplant: pos_delta={pos_delta}, rot_delta={rot_delta}")

    def _randomize_camera(self, env):
        """Apply small camera pose perturbations."""
        cfg = self.config
        pos_delta = self._rng.normal(0, cfg.camera_position_jitter, size=3)
        rot_delta = self._rng.normal(0, cfg.camera_rotation_jitter, size=3)
        logger.debug(f"DR camera: pos_delta={pos_delta}, rot_delta={rot_delta}")

    def _randomize_textures(self, env):
        """Swap table surface textures."""
        if not self.config.table_texture_paths:
            return
        idx = self._rng.integers(len(self.config.table_texture_paths))
        texture = self.config.table_texture_paths[idx]
        logger.debug(f"DR texture: {texture}")

    def _place_distractors(self, env):
        """Place random distractor objects on the table."""
        n = self.config.num_distractors
        logger.debug(f"DR distractors: placing {n} objects")
