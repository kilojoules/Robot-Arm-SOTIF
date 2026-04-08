"""Isaac Sim evaluators for policy evaluation under shader-based occlusion.

Provides two evaluator classes:

- ``IsaacSimEvaluator``: Single-environment evaluation. Steps one Isaac Lab
  env, sets Warp kernel shader uniforms per timestep, reads back camera
  images for the policy.

- ``IsaacBatchEvaluator``: GPU-parallel evaluation. Runs N Isaac Lab
  environments simultaneously (one per CMA-ES candidate), enabling
  an entire generation to be evaluated in the wall-clock time of a single
  episode.

Both require an NVIDIA GPU with Isaac Sim / Isaac Lab installed.
For CPU-only testing, use the mock objects in ``tests/test_isaac_evaluator.py``.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from adversarial_dust.evaluator import PolicyEvaluator
from adversarial_dust.occlusion_model import ContaminationMode

logger = logging.getLogger(__name__)

# Lazy imports — Isaac Sim is heavy and GPU-only.
_isaac_lab = None


def _ensure_isaac_lab():
    """Import Isaac Lab modules lazily."""
    global _isaac_lab
    if _isaac_lab is None:
        try:
            import omni.isaac.lab  # noqa: F401
            _isaac_lab = True
        except ImportError:
            raise ImportError(
                "Isaac Sim / Isaac Lab is required for IsaacSimEvaluator. "
                "Install via: pip install isaacsim-rl isaacsim-repl-extension-template"
            )


class IsaacSimEvaluator(PolicyEvaluator):
    """Single-environment Isaac Sim evaluator with shader-based occlusion.

    The evaluator manages an Isaac Lab ``ManagerBasedRLEnv``, applies
    contamination via Warp kernel uniforms on the camera render buffer,
    and feeds the resulting image to the policy.

    Args:
        isaac_config: Isaac Sim configuration (from config.py).
        env_config: Environment configuration (task, policy model, steps).
        policy: Manipulation policy with ``reset()`` / ``step()`` interface.
        dust_model: Occlusion model (SHADER or POST_PROCESS mode).
    """

    def __init__(self, isaac_config, env_config, policy, dust_model):
        self.isaac_config = isaac_config
        self.env_config = env_config
        self.policy = policy
        self.dust_model = dust_model
        self._sim = None
        self._shader_interface = None

    def _ensure_sim(self):
        """Lazy-init Isaac Sim simulation and camera."""
        if self._sim is not None:
            return
        _ensure_isaac_lab()
        # Actual Isaac Lab env creation would go here:
        # self._sim = ManagerBasedRLEnv(cfg=...)
        # self._camera = ...
        raise NotImplementedError(
            "Isaac Lab environment setup requires a concrete USD scene. "
            "See isaac_env_wrapper.py for the environment factory."
        )

    def _apply_contamination(self, params, timestep):
        """Set shader uniforms or apply post-process contamination."""
        if self.dust_model.contamination_mode == ContaminationMode.SHADER:
            shader_params = self.dust_model.get_shader_params(params, timestep)
            if self._shader_interface is not None:
                for key, value in shader_params.items():
                    self._shader_interface.set_param(key, value)
        # POST_PROCESS mode is handled after image readback in run_episode

    def run_episode(
        self,
        dust_params: Optional[np.ndarray],
        record: bool = False,
    ) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
        """Run a single episode in Isaac Sim.

        For SHADER mode, shader uniforms are set before each sim step.
        For POST_PROCESS mode, dust_model.apply() is called on the
        camera image after readback (fallback for non-shader models).
        """
        self._ensure_sim()

        # Reset simulation
        obs = self._sim.reset()
        instruction = self.env_config.task_name
        self.policy.reset(instruction)

        step_count = 0
        success = False
        clean_frames: List[np.ndarray] = []
        dirty_frames: List[np.ndarray] = []

        while step_count < self.env_config.max_episode_steps:
            # Apply shader contamination before render
            if dust_params is not None:
                self._apply_contamination(dust_params, step_count)

            # Step simulation (physics + render)
            self._sim.step()
            image = self._get_camera_image()

            if record:
                clean_frames.append(image.copy())

            # Post-process fallback for non-shader models
            if (
                dust_params is not None
                and self.dust_model.contamination_mode
                == ContaminationMode.POST_PROCESS
            ):
                image = self.dust_model.apply(image, dust_params, timestep=step_count)

            if record:
                dirty_frames.append(image.copy())

            raw_action, action = self.policy.step(image, instruction)
            action_array = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"].flatten(),
            ])
            self._sim.apply_action(action_array)

            step_count += 1

            if self._sim.is_success():
                success = True
                break

        return success, clean_frames, dirty_frames

    def _get_camera_image(self) -> np.ndarray:
        """Read back camera image from the GPU render buffer."""
        # Placeholder — actual implementation reads from Isaac Lab camera sensor
        raise NotImplementedError


class IsaacBatchEvaluator(PolicyEvaluator):
    """GPU-parallel evaluator: N environments with different shader params.

    Maps the CMA-ES population directly to parallel Isaac Lab environments.
    Each environment gets different contamination parameters while sharing
    the same policy weights, enabling an entire generation to be evaluated
    in the wall-clock time of a single episode.

    Args:
        isaac_config: Isaac Sim configuration (num_envs sets parallelism).
        env_config: Environment configuration.
        policy: Manipulation policy.
        dust_model: Occlusion model.
    """

    def __init__(self, isaac_config, env_config, policy, dust_model):
        self.isaac_config = isaac_config
        self.env_config = env_config
        self.policy = policy
        self.dust_model = dust_model
        self.num_envs = isaac_config.num_envs
        self._sim = None

    def _ensure_sim(self):
        """Lazy-init vectorized Isaac Sim environments."""
        if self._sim is not None:
            return
        _ensure_isaac_lab()
        raise NotImplementedError(
            "Vectorized Isaac Lab env setup requires USD scene. "
            "See isaac_env_wrapper.py for the environment factory."
        )

    def run_episode(
        self,
        dust_params: Optional[np.ndarray],
        record: bool = False,
    ) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
        """Single-candidate fallback — runs in env 0."""
        # For single-candidate evaluation, delegate to the serial path.
        results = self.batch_evaluate(
            [dust_params] if dust_params is not None else [None],
            n_episodes=1,
        )
        return results[0] >= 0.5, [], []

    def batch_evaluate(
        self,
        candidates: list,
        n_episodes: int,
    ) -> list:
        """Evaluate all candidates in parallel across GPU environments.

        Candidates are distributed across ``num_envs`` parallel
        environments. If ``len(candidates) > num_envs``, evaluation
        is batched in chunks.

        Args:
            candidates: List of parameter arrays (one per CMA-ES candidate).
            n_episodes: Episodes per candidate.

        Returns:
            List of mean success rates (one per candidate).
        """
        self._ensure_sim()

        pop_size = len(candidates)
        success_counts = np.zeros(pop_size)

        for ep in range(n_episodes):
            # Process candidates in chunks of num_envs
            for chunk_start in range(0, pop_size, self.num_envs):
                chunk_end = min(chunk_start + self.num_envs, pop_size)
                chunk = candidates[chunk_start:chunk_end]
                chunk_size = len(chunk)

                # Reset the first chunk_size envs
                self._sim.reset(env_ids=list(range(chunk_size)))

                # Set per-env shader params
                for env_idx, params in enumerate(chunk):
                    if params is not None:
                        shader_params = self.dust_model.get_shader_params(
                            np.asarray(params), timestep=0
                        )
                        self._set_env_shader(env_idx, shader_params)

                # Step all envs in lockstep
                for step in range(self.env_config.max_episode_steps):
                    self._sim.step()

                    # Update shader params with timestep
                    for env_idx, params in enumerate(chunk):
                        if params is not None:
                            shader_params = self.dust_model.get_shader_params(
                                np.asarray(params), timestep=step
                            )
                            self._set_env_shader(env_idx, shader_params)

                    # Read images and get policy actions
                    images = self._get_batch_images(chunk_size)
                    actions = self._batch_policy_step(images)
                    self._sim.apply_batch_actions(actions)

                # Collect success flags
                successes = self._sim.get_success_flags(chunk_size)
                success_counts[chunk_start:chunk_end] += successes

        return (success_counts / n_episodes).tolist()

    def _set_env_shader(self, env_idx: int, shader_params: dict):
        """Set shader uniforms on environment env_idx's camera."""
        raise NotImplementedError

    def _get_batch_images(self, n: int) -> list:
        """Read camera images from the first n environments."""
        raise NotImplementedError

    def _batch_policy_step(self, images: list) -> np.ndarray:
        """Get policy actions for a batch of images."""
        raise NotImplementedError
