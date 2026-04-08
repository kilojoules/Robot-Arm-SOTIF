"""Isaac Lab environment wrapper for manipulation tasks.

Wraps Isaac Lab's ``ManagerBasedRLEnv`` to provide the interface consumed
by ``IsaacSimEvaluator`` and ``IsaacBatchEvaluator``:

- Scene: table + basket + eggplant USD assets.
- Robot: WidowX-250 (converted from URDF).
- Camera: overhead or wrist-mounted, matching SimplerEnv pose/intrinsics.
- Success: eggplant centroid inside basket bounding volume.
- Action: 7-DOF (world_vector[3] + rot_axangle[3] + gripper[1])
          mapped to Isaac Lab's DifferentialIKController.

Requires Isaac Sim / Isaac Lab. For CPU-only testing, use MockIsaacEnv.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from adversarial_dust.config import IsaacSimConfig

logger = logging.getLogger(__name__)


@dataclass
class IsaacManipulationEnvCfg:
    """Configuration for the Isaac Lab manipulation environment.

    Encapsulates the scene USD paths, robot configuration, camera
    placement, and physics parameters needed to instantiate the env.
    """

    task_name: str = "widowx_put_eggplant_in_basket"
    num_envs: int = 1
    robot_usd_path: str = ""
    scene_usd_path: str = ""
    camera_width: int = 256
    camera_height: int = 256
    camera_prim_path: str = "/World/Camera"
    physics_dt: float = 1.0 / 120.0
    rendering_dt: float = 1.0 / 30.0
    headless: bool = True
    gpu_id: int = 0


class IsaacManipulationEnv:
    """Wrapper around Isaac Lab ManagerBasedRLEnv for manipulation.

    Provides the same interface as SimplerEnv for drop-in use by the
    evaluator classes.

    Note: This class requires Isaac Sim. The actual Isaac Lab integration
    will configure:

    - ``ArticulationCfg`` for the WidowX robot arm
    - ``RigidObjectCfg`` for the eggplant and basket
    - ``CameraCfg`` for the observation camera
    - ``DifferentialIKController`` for action mapping
    - ``EventTermCfg`` for success detection
    """

    def __init__(self, cfg: IsaacManipulationEnvCfg):
        self.cfg = cfg
        self._env = None
        self._step_count = 0

    def _create_env(self):
        """Create the Isaac Lab environment.

        This is where the USD scene, robot, camera, and task logic
        are configured. Deferred to first call of reset().
        """
        try:
            # Isaac Lab imports
            from omni.isaac.lab.envs import ManagerBasedRLEnv
        except ImportError:
            raise ImportError(
                "Isaac Lab is required. Install with: "
                "pip install isaacsim-rl isaacsim-repl-extension-template"
            )

        # TODO: Configure ManagerBasedRLEnvCfg with:
        #   - scene.robot = ArticulationCfg(usd_path=self.cfg.robot_usd_path)
        #   - scene.eggplant = RigidObjectCfg(...)
        #   - scene.basket = RigidObjectCfg(...)
        #   - scene.camera = CameraCfg(
        #       prim_path=self.cfg.camera_prim_path,
        #       width=self.cfg.camera_width,
        #       height=self.cfg.camera_height,
        #   )
        #   - actions = DifferentialIKControllerCfg(...)
        #   - terminations.success = EventTermCfg(
        #       func=check_eggplant_in_basket, ...)
        logger.info(
            f"Creating Isaac Lab env: {self.cfg.task_name} "
            f"(num_envs={self.cfg.num_envs}, headless={self.cfg.headless})"
        )
        raise NotImplementedError(
            "Isaac Lab environment creation requires concrete USD assets. "
            "Provide robot_usd_path and scene_usd_path in the config."
        )

    def reset(self, env_ids=None):
        """Reset environment(s). Returns observation dict."""
        if self._env is None:
            self._create_env()
        self._step_count = 0
        # Actual: obs = self._env.reset(env_ids)
        # Return image observation matching SimplerEnv format
        raise NotImplementedError

    def step(self, action):
        """Step the environment with a 7-DOF action array.

        Args:
            action: np.ndarray of shape (7,) or (num_envs, 7)
                [world_vector(3), rot_axangle(3), gripper(1)]

        Returns:
            (obs, reward, done, truncated, info)
        """
        self._step_count += 1
        # Actual: obs, reward, done, info = self._env.step(action_tensor)
        raise NotImplementedError

    def get_camera_image(self, env_idx: int = 0) -> np.ndarray:
        """Read camera image for environment env_idx.

        Returns:
            uint8 array of shape (H, W, 3).
        """
        # Actual: self._env.scene["camera"].data.output["rgb"][env_idx]
        raise NotImplementedError

    def get_batch_images(self, n: int) -> np.ndarray:
        """Read camera images from the first n environments.

        Returns:
            uint8 array of shape (n, H, W, 3).
        """
        raise NotImplementedError

    def is_success(self, env_idx: int = 0) -> bool:
        """Check if the task succeeded for environment env_idx."""
        raise NotImplementedError

    def get_success_flags(self, n: int) -> np.ndarray:
        """Get success flags for the first n environments.

        Returns:
            bool array of shape (n,).
        """
        raise NotImplementedError

    def get_language_instruction(self) -> str:
        """Return task instruction string."""
        return self.cfg.task_name

    def close(self):
        """Shutdown Isaac Sim."""
        if self._env is not None:
            self._env.close()
            self._env = None
